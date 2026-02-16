"""
Stream Diversion Specification Reader for IWFM.

This module reads the IWFM diversion specification file, which defines:
1. Diversion definitions with source nodes, delivery destinations,
   and column references for time-series data
2. Element groups (sets of elements for delivery)
3. Recharge zone destinations for recoverable losses

The file format supports two variants:
- 14-column format (legacy, without spill fields)
- 16-column format (with spill column and fraction)

Reference: Class_Diversion.f90 - Diversion_New()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    COMMENT_CHARS,
    is_comment_line as _is_comment_line,
    next_data_or_empty as _next_data_or_empty_f,
    strip_inline_comment as _parse_value_line,
)


# Delivery destination types (from Fortran source)
DEST_ELEMENT = 1
DEST_SUBREGION = 2
DEST_OUTSIDE = 3
DEST_ELEMENT_SET = 4


@dataclass
class DiversionSpec:
    """Specification for a single stream diversion.

    Attributes:
        id: Diversion ID
        stream_node: Stream node where diversion originates (0=outside model)
        max_diver_col: Column in diversions file for max diversion rate
        frac_max_diver: Fraction of max diversion to apply
        recv_loss_col: Column for recoverable loss (0=not used)
        frac_recv_loss: Fraction of recoverable loss
        non_recv_loss_col: Column for non-recoverable loss (0=not used)
        frac_non_recv_loss: Fraction of non-recoverable loss
        spill_col: Column for spill rates (0=not used, only in 16-col format)
        frac_spill: Fraction of spill (only in 16-col format)
        dest_type: Delivery destination type (1=element, 2=subregion, 3=outside, 4=elementset)
        dest_id: Destination element/subregion/set ID
        delivery_col: Column for delivery amounts (0=not used)
        frac_delivery: Fraction of delivery
        irrig_frac_col: Column for irrigation fraction
        adjustment_col: Column for adjustment data
        name: Diversion name (up to 20 chars)
    """
    id: int = 0
    stream_node: int = 0
    max_diver_col: int = 0
    frac_max_diver: float = 1.0
    recv_loss_col: int = 0
    frac_recv_loss: float = 0.0
    non_recv_loss_col: int = 0
    frac_non_recv_loss: float = 0.0
    spill_col: int = 0
    frac_spill: float = 0.0
    dest_type: int = DEST_OUTSIDE
    dest_id: int = 0
    delivery_col: int = 0
    frac_delivery: float = 1.0
    irrig_frac_col: int = 0
    adjustment_col: int = 0
    name: str = ""


@dataclass
class ElementGroup:
    """A group of elements used as a diversion delivery destination.

    Attributes:
        id: Group ID
        elements: List of element IDs in this group
    """
    id: int = 0
    elements: list[int] = field(default_factory=list)


@dataclass
class RechargeZoneDest:
    """Recharge zone destination for recoverable loss or spill.

    Attributes:
        diversion_id: ID of the associated diversion
        n_zones: Number of recharge zone destinations
        zone_ids: List of zone IDs
        zone_fractions: List of fractions for each zone
    """
    diversion_id: int = 0
    n_zones: int = 0
    zone_ids: list[int] = field(default_factory=list)
    zone_fractions: list[float] = field(default_factory=list)


@dataclass
class DiversionSpecConfig:
    """Complete diversion specification configuration.

    Attributes:
        n_diversions: Number of diversions
        diversions: List of diversion specifications
        n_element_groups: Number of element groups
        element_groups: List of element groups
        recharge_zones: List of recharge zone destinations (for recoverable loss)
        spill_zones: List of spill zone destinations (for spills)
        has_spills: Whether the file uses 16-column format with spills
    """
    n_diversions: int = 0
    diversions: list[DiversionSpec] = field(default_factory=list)
    n_element_groups: int = 0
    element_groups: list[ElementGroup] = field(default_factory=list)
    recharge_zones: list[RechargeZoneDest] = field(default_factory=list)
    spill_zones: list[RechargeZoneDest] = field(default_factory=list)
    has_spills: bool = False


class DiversionSpecReader:
    """Reader for IWFM diversion specification files.

    Supports both 14-column (legacy) and 16-column (with spills) formats.
    Auto-detects the format based on the number of columns in the first
    diversion data line.
    """

    def __init__(self) -> None:
        self._line_num = 0

    def read(self, filepath: Path | str) -> DiversionSpecConfig:
        """Read diversion specification file.

        Args:
            filepath: Path to the diversion spec file

        Returns:
            DiversionSpecConfig with all diversion data
        """
        filepath = Path(filepath)
        config = DiversionSpecConfig()
        self._line_num = 0

        with open(filepath, "r") as f:
            # NDiver (number of diversions)
            ndiver_str = self._next_data_or_empty(f)
            if not ndiver_str:
                return config
            config.n_diversions = int(ndiver_str)

            if config.n_diversions <= 0:
                return config

            # Read diversion specifications
            # Auto-detect format from first line
            first_line = self._next_data_line(f)
            parts = first_line.split()

            # Determine format: 16+ columns = with spills, 14+ = without
            # With name at end, total columns could be 15 (14+name) or 17 (16+name)
            # We detect based on numeric field count
            n_numeric = self._count_numeric_fields(parts)
            config.has_spills = n_numeric >= 16

            # Parse first diversion
            config.diversions.append(
                self._parse_diversion_line(parts, config.has_spills)
            )

            # Parse remaining diversions
            for _ in range(config.n_diversions - 1):
                line = self._next_data_line(f)
                parts = line.split()
                config.diversions.append(
                    self._parse_diversion_line(parts, config.has_spills)
                )

            # Element groups
            ngroup_str = self._next_data_or_empty(f)
            if ngroup_str:
                config.n_element_groups = int(ngroup_str)

                for _ in range(config.n_element_groups):
                    group = self._read_element_group(f)
                    config.element_groups.append(group)

            # Recharge zones — one entry per diversion
            for div in config.diversions:
                try:
                    rz = self._read_recharge_zone(f, div.id)
                    config.recharge_zones.append(rz)
                except (FileFormatError, StopIteration):
                    break

            # Spill zones (only if has_spills format) — one entry per diversion
            if config.has_spills:
                for div in config.diversions:
                    try:
                        sz = self._read_recharge_zone(f, div.id)
                        config.spill_zones.append(sz)
                    except (FileFormatError, StopIteration):
                        break

        return config

    def _count_numeric_fields(self, parts: list[str]) -> int:
        """Count the number of numeric fields in a line."""
        count = 0
        for p in parts:
            try:
                float(p)
                count += 1
            except ValueError:
                break
        return count

    def _parse_diversion_line(
        self, parts: list[str], has_spills: bool
    ) -> DiversionSpec:
        """Parse a single diversion specification line.

        Args:
            parts: Split line tokens
            has_spills: Whether this is 16-column format

        Returns:
            DiversionSpec with parsed data
        """
        spec = DiversionSpec()

        try:
            idx = 0
            spec.id = int(parts[idx]); idx += 1
            spec.stream_node = int(parts[idx]); idx += 1
            spec.max_diver_col = int(parts[idx]); idx += 1
            spec.frac_max_diver = float(parts[idx]); idx += 1
            spec.recv_loss_col = int(parts[idx]); idx += 1
            spec.frac_recv_loss = float(parts[idx]); idx += 1
            spec.non_recv_loss_col = int(parts[idx]); idx += 1
            spec.frac_non_recv_loss = float(parts[idx]); idx += 1

            if has_spills:
                spec.spill_col = int(parts[idx]); idx += 1
                spec.frac_spill = float(parts[idx]); idx += 1

            spec.dest_type = int(parts[idx]); idx += 1
            spec.dest_id = int(parts[idx]); idx += 1
            spec.delivery_col = int(parts[idx]); idx += 1
            spec.frac_delivery = float(parts[idx]); idx += 1
            spec.irrig_frac_col = int(parts[idx]); idx += 1
            spec.adjustment_col = int(parts[idx]); idx += 1

            # Name is optional, remainder of line
            if idx < len(parts):
                spec.name = " ".join(parts[idx:])
        except (IndexError, ValueError):
            # Partial parse is acceptable
            pass

        return spec

    @staticmethod
    def _strip_inline_comment(val: str) -> str:
        """Strip inline comment char (/) from a value."""
        pos = val.find("/")
        if pos >= 0:
            val = val[:pos]
        return val.strip()

    def _read_element_group(self, f: TextIO) -> ElementGroup:
        """Read a single element group.

        Format:
            GroupID  NElements  FirstElementID
            ElementID (one per subsequent line)
        """
        header_line = self._next_data_line(f)
        parts = header_line.split()

        group = ElementGroup()
        group.id = int(self._strip_inline_comment(parts[0]))
        n_elements = int(self._strip_inline_comment(parts[1]))

        # First element may be on the header line
        if len(parts) >= 3:
            elem_str = self._strip_inline_comment(parts[2])
            if elem_str:
                group.elements.append(int(elem_str))
            remaining = n_elements - 1
        else:
            remaining = n_elements

        for _ in range(remaining):
            elem_line = self._next_data_line(f)
            elem_str = self._strip_inline_comment(elem_line.split()[0])
            group.elements.append(int(elem_str))

        return group

    def _read_recharge_zone(self, f: TextIO, diversion_id: int) -> RechargeZoneDest:
        """Read recharge zone destination data for a diversion.

        Format (LossDestination_New in Fortran):
            ID  NERELS  first_IERELS  first_FERELS
            IERELS  FERELS  (for each remaining element)

        When NERELS=0, the entire entry is on one line:
            ID  0  0  0
        """
        rz = RechargeZoneDest(diversion_id=diversion_id)

        header_line = self._next_data_line(f)
        parts = header_line.split()

        try:
            # Parse: ID  NERELS  [first_IERELS  first_FERELS]
            rz.diversion_id = int(self._strip_inline_comment(parts[0]))
            rz.n_zones = int(self._strip_inline_comment(parts[1]))

            if rz.n_zones > 0 and len(parts) >= 4:
                # First element is on the header line
                rz.zone_ids.append(int(self._strip_inline_comment(parts[2])))
                rz.zone_fractions.append(float(self._strip_inline_comment(parts[3])))

                # Read remaining NERELS-1 elements
                for _ in range(rz.n_zones - 1):
                    line = self._next_data_line(f)
                    elem_parts = line.split()
                    rz.zone_ids.append(int(self._strip_inline_comment(elem_parts[0])))
                    if len(elem_parts) > 1:
                        rz.zone_fractions.append(float(self._strip_inline_comment(elem_parts[1])))
                    else:
                        rz.zone_fractions.append(1.0)
        except (IndexError, ValueError):
            pass

        return rz

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


def read_diversion_spec(filepath: Path | str) -> DiversionSpecConfig:
    """Read IWFM diversion specification file.

    Args:
        filepath: Path to the diversion spec file

    Returns:
        DiversionSpecConfig with all diversion data
    """
    reader = DiversionSpecReader()
    return reader.read(filepath)
