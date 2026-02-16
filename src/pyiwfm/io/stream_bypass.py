"""
Stream Bypass Specification Reader for IWFM.

This module reads the IWFM bypass specification file, which defines
conversion factors for flow and bypass values, bypass definitions with
source/destination nodes, rating tables, and recoverable/non-recoverable
loss fractions. Bypass destination types are 1 (stream node) or 2 (lake).

Reference: Class_Bypass.f90 - Bypass_New()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    COMMENT_CHARS,
    is_comment_line as _is_comment_line,
    next_data_or_empty as _next_data_or_empty_f,
    strip_inline_comment as _parse_value_line,
)


# Bypass destination types
BYPASS_DEST_STREAM = 1
BYPASS_DEST_LAKE = 2


@dataclass
class BypassRatingTable:
    """Rating table for a bypass (inline definition).

    Attributes:
        flows: Array of flow values
        fractions: Array of bypass fraction at each flow level
    """
    flows: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    fractions: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))


@dataclass
class BypassSpec:
    """Specification for a single bypass.

    Attributes:
        id: Bypass ID
        export_stream_node: Stream node where bypass originates (0=outside model)
        dest_type: Destination type (1=stream node, 2=lake)
        dest_id: Destination stream node ID or lake ID
        rating_table_col: Column in diversions file for rating.
            Positive means pre-defined rating from diversions file,
            negative means ``abs(col)`` inline rating table points,
            zero means no rating table.
        frac_recoverable: Fraction of bypass that is recoverable loss
        frac_non_recoverable: Fraction of bypass that is non-recoverable loss
        name: Bypass name (up to 20 chars)
        inline_rating: Inline rating table (if rating_table_col < 0)
    """
    id: int = 0
    export_stream_node: int = 0
    dest_type: int = BYPASS_DEST_STREAM
    dest_id: int = 0
    rating_table_col: int = 0
    frac_recoverable: float = 0.0
    frac_non_recoverable: float = 0.0
    name: str = ""
    inline_rating: BypassRatingTable | None = None


@dataclass
class BypassSeepageZone:
    """Seepage/recharge zone for a bypass.

    Attributes:
        bypass_id: ID of the associated bypass
        n_elements: Number of elements receiving recharge
        element_ids: List of element IDs
        element_fractions: List of fractions for each element
    """
    bypass_id: int = 0
    n_elements: int = 0
    element_ids: list[int] = field(default_factory=list)
    element_fractions: list[float] = field(default_factory=list)


@dataclass
class BypassSpecConfig:
    """Complete bypass specification configuration.

    Attributes:
        n_bypasses: Number of bypasses
        flow_factor: Flow conversion factor for stream flows
        flow_time_unit: Time unit for stream flows
        bypass_factor: Flow conversion factor for bypass values
        bypass_time_unit: Time unit for bypass values
        bypasses: List of bypass specifications
        seepage_zones: List of seepage zone destinations (per bypass)
    """
    n_bypasses: int = 0
    flow_factor: float = 1.0
    flow_time_unit: str = ""
    bypass_factor: float = 1.0
    bypass_time_unit: str = ""
    bypasses: list[BypassSpec] = field(default_factory=list)
    seepage_zones: list[BypassSeepageZone] = field(default_factory=list)


class BypassSpecReader:
    """Reader for IWFM bypass specification files.

    The bypass file defines flow routing that bypasses portions of
    the stream network, with optional inline rating tables.
    """

    def __init__(self) -> None:
        self._line_num = 0

    def read(self, filepath: Path | str) -> BypassSpecConfig:
        """Read bypass specification file.

        Args:
            filepath: Path to the bypass spec file

        Returns:
            BypassSpecConfig with all bypass data
        """
        filepath = Path(filepath)
        config = BypassSpecConfig()
        self._line_num = 0

        with open(filepath, "r") as f:
            # NBypass
            nbypass_str = self._next_data_or_empty(f)
            if not nbypass_str:
                return config
            config.n_bypasses = int(nbypass_str)

            if config.n_bypasses <= 0:
                return config

            # Flow conversion factor
            factor_str = self._next_data_or_empty(f)
            if factor_str:
                config.flow_factor = float(factor_str)

            # Stream flow time unit
            config.flow_time_unit = self._next_data_or_empty(f)

            # Bypass conversion factor
            bypass_factor_str = self._next_data_or_empty(f)
            if bypass_factor_str:
                config.bypass_factor = float(bypass_factor_str)

            # Bypass time unit
            config.bypass_time_unit = self._next_data_or_empty(f)

            # Read bypass specifications
            for _ in range(config.n_bypasses):
                bypass = self._read_bypass(f, config.flow_factor)
                config.bypasses.append(bypass)

            # Read seepage/recharge zones — one entry per bypass
            for bp in config.bypasses:
                try:
                    sz = self._read_seepage_zone(f, bp.id)
                    config.seepage_zones.append(sz)
                except (FileFormatError, StopIteration):
                    break

        return config

    def _read_bypass(self, f: TextIO, flow_factor: float) -> BypassSpec:
        """Read a single bypass specification.

        Args:
            f: Open file handle
            flow_factor: Flow conversion factor for rating table values

        Returns:
            BypassSpec with parsed data
        """
        line = self._next_data_line(f)
        parts = line.split()

        spec = BypassSpec()
        try:
            idx = 0
            spec.id = int(parts[idx]); idx += 1
            spec.export_stream_node = int(parts[idx]); idx += 1
            spec.dest_type = int(parts[idx]); idx += 1
            spec.dest_id = int(parts[idx]); idx += 1
            spec.rating_table_col = int(parts[idx]); idx += 1
            spec.frac_recoverable = float(parts[idx]); idx += 1
            spec.frac_non_recoverable = float(parts[idx]); idx += 1

            # Name is optional — take text up to inline comment (/)
            if idx < len(parts):
                name_parts = []
                for p in parts[idx:]:
                    if p.startswith("/"):
                        break
                    name_parts.append(p)
                spec.name = " ".join(name_parts)[:20].strip()
        except (IndexError, ValueError):
            pass

        # Read inline rating table if rating_table_col < 0
        if spec.rating_table_col < 0:
            n_points = abs(spec.rating_table_col)
            flows = []
            fractions = []

            for _ in range(n_points):
                rt_line = self._next_data_line(f)
                rt_parts = rt_line.split()
                if len(rt_parts) >= 2:
                    flows.append(float(rt_parts[0]) * flow_factor)
                    fractions.append(float(rt_parts[1]))

            spec.inline_rating = BypassRatingTable(
                flows=np.array(flows, dtype=np.float64),
                fractions=np.array(fractions, dtype=np.float64),
            )

        return spec

    def _read_seepage_zone(self, f: TextIO, bypass_id: int) -> BypassSeepageZone:
        """Read seepage/recharge zone data for a bypass.

        Format (LossDestination_New in Fortran):
            ID  NERELS  first_IERELS  first_FERELS
            IERELS  FERELS  (for remaining elements)
        """
        sz = BypassSeepageZone(bypass_id=bypass_id)

        line = self._next_data_line(f)
        parts = line.split()

        try:
            sz.bypass_id = int(parts[0])
            sz.n_elements = int(parts[1])

            if sz.n_elements > 0 and len(parts) >= 4:
                sz.element_ids.append(int(parts[2]))
                sz.element_fractions.append(float(parts[3]))

                for _ in range(sz.n_elements - 1):
                    elem_line = self._next_data_line(f)
                    elem_parts = elem_line.split()
                    sz.element_ids.append(int(elem_parts[0]))
                    if len(elem_parts) > 1:
                        sz.element_fractions.append(float(elem_parts[1]))
                    else:
                        sz.element_fractions.append(1.0)
        except (IndexError, ValueError):
            pass

        return sz

    def _next_data_or_empty(self, f: TextIO) -> str:
        """Return next data value, or empty string."""
        lc = [self._line_num]
        val = _next_data_or_empty_f(f, lc)
        self._line_num = lc[0]
        return val

    def _next_data_line(self, f: TextIO) -> str:
        """Return the next non-comment data line."""
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            return line.strip()
        raise FileFormatError("Unexpected end of file", line_number=self._line_num)


def read_bypass_spec(filepath: Path | str) -> BypassSpecConfig:
    """Read IWFM bypass specification file.

    Args:
        filepath: Path to the bypass spec file

    Returns:
        BypassSpecConfig with all bypass data
    """
    reader = BypassSpecReader()
    return reader.read(filepath)
