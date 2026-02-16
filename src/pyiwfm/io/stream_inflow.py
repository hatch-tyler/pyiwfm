"""
Stream Inflow Reader for IWFM.

This module reads the IWFM stream inflow file header, which maps
inflow time-series columns to stream nodes. The actual time-series
data follows in standard IWFM time-series format.

The header contains:
1. Conversion factor for all inflow values
2. Time unit
3. Number of inflow series
4. Per-series: [InflowID] StreamNodeID mapping

Reference: Class_StrmInflow.f90 - New()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    COMMENT_CHARS,
    is_comment_line as _is_comment_line,
    strip_inline_comment as _parse_value_line,
)


@dataclass
class InflowSpec:
    """Specification for a single inflow point.

    Attributes:
        inflow_id: Inflow identifier (may be auto-numbered)
        stream_node: Stream node receiving inflow (0=no inflow)
    """
    inflow_id: int = 0
    stream_node: int = 0


@dataclass
class InflowConfig:
    """Configuration from stream inflow file header.

    Attributes:
        conversion_factor: Multiplication factor for all inflow values
        time_unit: Time unit string (e.g., "MONTH", "DAY")
        n_inflows: Number of inflow time series
        inflow_specs: List of inflow specifications (column-to-node mapping)
    """
    conversion_factor: float = 1.0
    time_unit: str = ""
    n_inflows: int = 0
    inflow_specs: list[InflowSpec] = field(default_factory=list)

    @property
    def inflow_nodes(self) -> list[int]:
        """Return list of stream nodes receiving inflows (non-zero only)."""
        return [s.stream_node for s in self.inflow_specs if s.stream_node > 0]


class InflowReader:
    """Reader for IWFM stream inflow file header.

    Reads only the header (conversion factor, time unit, number of series,
    and per-series node mapping). Does not parse the time-series data.
    """

    def __init__(self) -> None:
        self._line_num = 0

    def read(self, filepath: Path | str) -> InflowConfig:
        """Read stream inflow file header.

        Args:
            filepath: Path to the stream inflow file

        Returns:
            InflowConfig with header data
        """
        filepath = Path(filepath)
        config = InflowConfig()
        self._line_num = 0

        with open(filepath, "r") as f:
            # Conversion factor
            factor_str = self._next_data_or_empty(f)
            if factor_str:
                config.conversion_factor = float(factor_str)

            # Time unit
            config.time_unit = self._next_data_or_empty(f)

            # Number of inflow series
            ninflow_str = self._next_data_or_empty(f)
            if not ninflow_str:
                return config
            config.n_inflows = int(ninflow_str)

            if config.n_inflows <= 0:
                return config

            # Read inflow specifications
            for i in range(config.n_inflows):
                line = self._next_data_line(f)
                parts = line.split()

                spec = InflowSpec()
                if len(parts) >= 2:
                    # Two-column format: InflowID StreamNodeID
                    spec.inflow_id = int(parts[0])
                    spec.stream_node = int(parts[1])
                elif len(parts) == 1:
                    # Single-column format: StreamNodeID only (auto-numbered)
                    spec.inflow_id = i + 1
                    spec.stream_node = int(parts[0])

                config.inflow_specs.append(spec)

        return config

    def _next_data_or_empty(self, f: TextIO) -> str:
        """Return next data value, or empty string."""
        for line in f:
            self._line_num += 1
            if line and line[0] in COMMENT_CHARS:
                continue
            value, _ = _parse_value_line(line)
            return value
        return ""

    def _next_data_line(self, f: TextIO) -> str:
        """Return the next non-comment data line."""
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            return line.strip()
        raise FileFormatError("Unexpected end of file", line_number=self._line_num)


def read_stream_inflow(filepath: Path | str) -> InflowConfig:
    """Read IWFM stream inflow file header.

    Args:
        filepath: Path to the stream inflow file

    Returns:
        InflowConfig with header data
    """
    reader = InflowReader()
    return reader.read(filepath)
