"""
Groundwater Tile Drain and Sub-Irrigation Reader for IWFM.

This module reads the IWFM tile drain/sub-irrigation file, which contains:
1. Tile drain specifications (node, elevation, conductance, destination)
2. Sub-irrigation specifications (node, elevation, conductance)

Both are in the same input file: tile drains first, then sub-irrigation.
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


# Destination types for tile drains
TD_DEST_OUTSIDE = 1    # Flow goes outside model domain
TD_DEST_STREAM = 2     # Flow goes to a stream node


@dataclass
class TileDrainSpec:
    """Tile drain specification.

    Attributes:
        id: Tile drain ID
        gw_node: GW node ID
        elevation: Drain elevation
        conductance: Drain conductance
        dest_type: Destination type (1=outside, 2=stream node)
        dest_id: Destination ID (stream node number if dest_type=2)
    """
    id: int
    gw_node: int
    elevation: float
    conductance: float
    dest_type: int = TD_DEST_OUTSIDE
    dest_id: int = 0


@dataclass
class SubIrrigationSpec:
    """Sub-irrigation (subsurface irrigation) specification.

    Attributes:
        id: Sub-irrigation ID
        gw_node: GW node ID
        elevation: Sub-irrigation elevation
        conductance: Conductance
    """
    id: int
    gw_node: int
    elevation: float
    conductance: float


@dataclass
class TileDrainConfig:
    """Complete tile drain and sub-irrigation configuration.

    Attributes:
        version: File format version (e.g., "4.0")
        n_drains: Number of tile drains
        drain_height_factor: Height conversion factor for drains
        drain_conductance_factor: Conductance conversion factor for drains
        drain_time_unit: Time unit for drain conductance
        tile_drains: List of tile drain specifications

        n_sub_irrigation: Number of sub-irrigation locations
        subirig_height_factor: Height conversion factor
        subirig_conductance_factor: Conductance conversion factor
        subirig_time_unit: Time unit for sub-irrigation conductance
        sub_irrigations: List of sub-irrigation specifications
    """
    version: str = ""
    # Tile drains
    n_drains: int = 0
    drain_height_factor: float = 1.0
    drain_conductance_factor: float = 1.0
    drain_time_unit: str = ""
    tile_drains: list[TileDrainSpec] = field(default_factory=list)

    # Sub-irrigation
    n_sub_irrigation: int = 0
    subirig_height_factor: float = 1.0
    subirig_conductance_factor: float = 1.0
    subirig_time_unit: str = ""
    sub_irrigations: list[SubIrrigationSpec] = field(default_factory=list)


class TileDrainReader:
    """Reader for IWFM tile drain/sub-irrigation file.

    The file contains tile drains first, then sub-irrigation data.
    """

    def __init__(self) -> None:
        self._line_num = 0
        self._pushed_back: str | None = None

    def read(self, filepath: Path | str) -> TileDrainConfig:
        """Read tile drain and sub-irrigation file.

        Args:
            filepath: Path to the tile drain file

        Returns:
            TileDrainConfig with all data
        """
        filepath = Path(filepath)
        config = TileDrainConfig()
        self._line_num = 0
        self._pushed_back = None

        with open(filepath, "r") as f:
            # Version header (e.g., #4.0)
            config.version = self._read_version(f)

            # --- Tile Drains Section ---

            # NDrain
            ndrain_str = self._next_data_or_empty(f)
            config.n_drains = int(ndrain_str) if ndrain_str else 0

            if config.n_drains > 0:
                # FactH
                config.drain_height_factor = float(self._next_data_or_empty(f))
                # FactCDC
                config.drain_conductance_factor = float(self._next_data_or_empty(f))
                # TimeUnit
                config.drain_time_unit = self._next_data_or_empty(f)

                # Read NDrain rows: ID, GWNode, Elevation, Conductance, DestType, Dest
                for _ in range(config.n_drains):
                    line = self._next_data_line(f)
                    parts = line.split()
                    if len(parts) < 6:
                        continue

                    config.tile_drains.append(TileDrainSpec(
                        id=int(parts[0]),
                        gw_node=int(parts[1]),
                        elevation=float(parts[2]) * config.drain_height_factor,
                        conductance=float(parts[3]) * config.drain_conductance_factor,
                        dest_type=int(parts[4]),
                        dest_id=int(parts[5]),
                    ))

            # --- Sub-Irrigation Section ---

            # NSubIrig
            nsubirig_str = self._next_data_or_empty(f)
            config.n_sub_irrigation = int(nsubirig_str) if nsubirig_str else 0

            if config.n_sub_irrigation > 0:
                # FactH
                config.subirig_height_factor = float(self._next_data_or_empty(f))
                # FactCDC
                config.subirig_conductance_factor = float(self._next_data_or_empty(f))
                # TimeUnit
                config.subirig_time_unit = self._next_data_or_empty(f)

                # Read NSubIrig rows: ID, GWNode, Elevation, Conductance
                for _ in range(config.n_sub_irrigation):
                    line = self._next_data_line(f)
                    parts = line.split()
                    if len(parts) < 4:
                        continue

                    config.sub_irrigations.append(SubIrrigationSpec(
                        id=int(parts[0]),
                        gw_node=int(parts[1]),
                        elevation=float(parts[2]) * config.subirig_height_factor,
                        conductance=float(parts[3]) * config.subirig_conductance_factor,
                    ))

        return config

    def _read_version(self, f: TextIO) -> str:
        """Read the version header (e.g., '#4.0')."""
        for line in f:
            self._line_num += 1
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped[1:].strip()
            if line[0] in COMMENT_CHARS:
                continue
            # Not a version line — this is the first data line (no version header)
            # We need to "push back" this line. Since we can't unseek easily,
            # parse it as NDrain directly.
            value, _ = _parse_value_line(line)
            # Store on instance for the caller to use
            self._pushed_back = value
            return ""
        return ""

    def _next_data_or_empty(self, f: TextIO) -> str:
        """Return next data value, or empty string."""
        # Check for pushed-back value from _read_version
        if hasattr(self, "_pushed_back") and self._pushed_back is not None:
            val = self._pushed_back
            self._pushed_back = None
            return val
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


def read_gw_tiledrain(filepath: Path | str) -> TileDrainConfig:
    """Read IWFM tile drain/sub-irrigation file.

    Args:
        filepath: Path to the tile drain file

    Returns:
        TileDrainConfig with all data
    """
    reader = TileDrainReader()
    return reader.read(filepath)
