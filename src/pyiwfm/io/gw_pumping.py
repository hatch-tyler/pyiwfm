"""
Groundwater Pumping Reader for IWFM.

This module reads the IWFM groundwater pumping files, which define:
1. Well specifications (location, perforation intervals, radius)
2. Well pumping specifications (rates, distribution, destinations)
3. Element-based pumping specifications
4. Element groups for pumping destinations
5. Time series pumping data references

The pumping main file references sub-files for wells, element pumping,
and a time series data file containing the actual pumping rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    COMMENT_CHARS,
)
from pyiwfm.io.iwfm_reader import (
    is_comment_line as _is_comment_line,
)
from pyiwfm.io.iwfm_reader import (
    next_data_or_empty as _next_data_or_empty,
)
from pyiwfm.io.iwfm_reader import (
    resolve_path as _resolve_path_f,
)

# =============================================================================
# Pumping distribution methods
# =============================================================================

DIST_USER_FRAC = 0  # Use user-specified fractions
DIST_TOTAL_AREA = 1  # Distribute by total area
DIST_AG_URB_AREA = 2  # Distribute by developed area
DIST_AG_AREA = 3  # Distribute by agricultural area
DIST_URB_AREA = 4  # Distribute by urban area

# Destination types
DEST_SAME_ELEMENT = -1  # Deliver to same element as well
DEST_OUTSIDE = 0  # Deliver outside model domain
DEST_ELEMENT = 1  # Deliver to specified element
DEST_SUBREGION = 2  # Deliver to specified subregion
DEST_ELEM_GROUP = 3  # Deliver to element group


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class WellSpec:
    """Well physical specification.

    Attributes:
        id: Well ID
        x: X coordinate
        y: Y coordinate
        radius: Well radius (half of diameter)
        perf_top: Perforation top elevation
        perf_bottom: Perforation bottom elevation
        name: Well name/description (from / delimiter in spec file)
    """

    id: int
    x: float
    y: float
    radius: float = 0.0
    perf_top: float = 0.0
    perf_bottom: float = 0.0
    name: str = ""


@dataclass
class WellPumpingSpec:
    """Well pumping specification.

    Attributes:
        well_id: Well ID (must match a WellSpec)
        pump_column: Column in time series pumping file
        pump_fraction: Fraction of pumping at this column
        dist_method: Distribution method (0-4)
        dest_type: Destination type (-1, 0, 1, 2, 3)
        dest_id: Destination ID (element, subregion, or group)
        irig_frac_column: Column for irrigation fraction
        adjust_column: Column for supply adjustment
        pump_max_column: Column for maximum pumping
        pump_max_fraction: Fraction of maximum pumping
    """

    well_id: int
    pump_column: int = 0
    pump_fraction: float = 1.0
    dist_method: int = DIST_USER_FRAC
    dest_type: int = DEST_SAME_ELEMENT
    dest_id: int = 0
    irig_frac_column: int = 0
    adjust_column: int = 0
    pump_max_column: int = 0
    pump_max_fraction: float = 0.0


@dataclass
class ElementPumpingSpec:
    """Element-based pumping specification.

    Attributes:
        element_id: Element ID
        pump_column: Column in time series pumping file
        pump_fraction: Fraction of pumping at this column
        dist_method: Distribution method (0-4)
        layer_factors: Layer distribution factors
        dest_type: Destination type (-1, 0, 1, 2, 3)
        dest_id: Destination ID
        irig_frac_column: Column for irrigation fraction
        adjust_column: Column for supply adjustment
        pump_max_column: Column for maximum pumping
        pump_max_fraction: Fraction of maximum pumping
    """

    element_id: int
    pump_column: int = 0
    pump_fraction: float = 1.0
    dist_method: int = DIST_USER_FRAC
    layer_factors: list[float] = field(default_factory=list)
    dest_type: int = DEST_SAME_ELEMENT
    dest_id: int = 0
    irig_frac_column: int = 0
    adjust_column: int = 0
    pump_max_column: int = 0
    pump_max_fraction: float = 0.0


@dataclass
class ElementGroup:
    """Group of elements for pumping destination.

    Attributes:
        id: Group ID
        elements: List of element IDs in this group
    """

    id: int
    elements: list[int] = field(default_factory=list)


@dataclass
class PumpingConfig:
    """Complete pumping configuration.

    Attributes:
        version: File format version
        well_file: Path to well specification file
        elem_pump_file: Path to element pumping specification file
        ts_data_file: Path to time series pumping data file
        output_file: Path to pumping output file

        well_specs: Well physical specifications
        well_pumping_specs: Well pumping specifications
        well_groups: Element groups for well destinations
        factor_xy: Coordinate conversion factor
        factor_radius: Radius conversion factor
        factor_length: Perforation length conversion factor

        elem_pumping_specs: Element pumping specifications
        elem_groups: Element groups for element pumping destinations

        pump_factor: Pumping rate conversion factor
    """

    version: str = ""
    well_file: Path | None = None
    elem_pump_file: Path | None = None
    ts_data_file: Path | None = None
    output_file: Path | None = None

    # Well data
    well_specs: list[WellSpec] = field(default_factory=list)
    well_pumping_specs: list[WellPumpingSpec] = field(default_factory=list)
    well_groups: list[ElementGroup] = field(default_factory=list)
    factor_xy: float = 1.0
    factor_radius: float = 1.0
    factor_length: float = 1.0

    # Element pumping data
    elem_pumping_specs: list[ElementPumpingSpec] = field(default_factory=list)
    elem_groups: list[ElementGroup] = field(default_factory=list)

    # Conversion
    pump_factor: float = 1.0

    @property
    def n_wells(self) -> int:
        return len(self.well_specs)

    @property
    def n_elem_pumping(self) -> int:
        return len(self.elem_pumping_specs)


class PumpingReader:
    """Reader for IWFM pumping files.

    The pumping system reads from a main file that references:
    - Well specification file
    - Element pumping specification file
    - Time series pumping data file
    """

    def __init__(self) -> None:
        self._line_num = 0

    def read(
        self, filepath: Path | str, base_dir: Path | None = None, n_layers: int = 1
    ) -> PumpingConfig:
        """Read pumping main file and referenced sub-files.

        Args:
            filepath: Path to main pumping file
            base_dir: Base directory for resolving relative paths
            n_layers: Number of aquifer layers (needed for element pumping)

        Returns:
            PumpingConfig with all pumping data
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        config = PumpingConfig()
        self._line_num = 0

        with open(filepath) as f:
            # Version
            config.version = self._read_version(f)

            # Well specification file
            well_path = _next_data_or_empty(f)
            if well_path:
                config.well_file = _resolve_path_f(base_dir, well_path)

            # Element pumping specification file
            elem_path = _next_data_or_empty(f)
            if elem_path:
                config.elem_pump_file = _resolve_path_f(base_dir, elem_path)

            # Time series pumping data file
            ts_path = _next_data_or_empty(f)
            if ts_path:
                config.ts_data_file = _resolve_path_f(base_dir, ts_path)

            # Output file (backward compatible)
            out_path = _next_data_or_empty(f)
            if out_path:
                config.output_file = _resolve_path_f(base_dir, out_path)

        # Read well specification file
        if config.well_file and config.well_file.exists():
            self._read_well_file(config.well_file, config)

        # Read element pumping file
        if config.elem_pump_file and config.elem_pump_file.exists():
            self._read_elem_pump_file(config.elem_pump_file, config, n_layers)

        return config

    def _read_well_file(self, filepath: Path, config: PumpingConfig) -> None:
        """Read well specification file."""
        self._line_num = 0
        with open(filepath) as f:
            # NWell
            n_wells = int(_next_data_or_empty(f))
            if n_wells <= 0:
                return

            # FactXY
            config.factor_xy = float(_next_data_or_empty(f))
            # FactR
            config.factor_radius = float(_next_data_or_empty(f))
            # FactLT
            config.factor_length = float(_next_data_or_empty(f))

            # Read structural data for each well: ID, X, Y, Radius, PerfTop, PerfBottom [/ Name]
            for _ in range(n_wells):
                line = self._next_data_line(f)
                # Extract name from / delimiter if present
                name = ""
                if "/" in line:
                    slash_idx = line.index("/")
                    name = line[slash_idx + 1 :].strip()
                    line = line[:slash_idx].strip()
                parts = line.split()
                if len(parts) < 6:
                    continue

                config.well_specs.append(
                    WellSpec(
                        id=int(float(parts[0])),
                        x=float(parts[1]) * config.factor_xy,
                        y=float(parts[2]) * config.factor_xy,
                        radius=float(parts[3]) / 2.0 * config.factor_radius,
                        perf_top=float(parts[4]) * config.factor_length,
                        perf_bottom=float(parts[5]) * config.factor_length,
                        name=name,
                    )
                )

            # Read pumping specifications: 10 values per well
            for _ in range(n_wells):
                line = self._next_data_line(f)
                parts = line.split()
                if len(parts) < 10:
                    continue

                config.well_pumping_specs.append(
                    WellPumpingSpec(
                        well_id=int(float(parts[0])),
                        pump_column=int(float(parts[1])),
                        pump_fraction=float(parts[2]),
                        dist_method=int(float(parts[3])),
                        dest_type=int(float(parts[4])),
                        dest_id=int(float(parts[5])),
                        irig_frac_column=int(float(parts[6])),
                        adjust_column=int(float(parts[7])),
                        pump_max_column=int(float(parts[8])),
                        pump_max_fraction=float(parts[9]),
                    )
                )

            # Read element groups
            config.well_groups = self._read_element_groups(f)

    def _read_elem_pump_file(self, filepath: Path, config: PumpingConfig, n_layers: int) -> None:
        """Read element pumping specification file."""
        self._line_num = 0
        with open(filepath) as f:
            # NSink
            n_sinks = int(_next_data_or_empty(f))
            if n_sinks <= 0:
                return

            # Read each element pump: (10 + NLayers) values
            for _ in range(n_sinks):
                line = self._next_data_line(f)
                parts = line.split()
                expected = 10 + n_layers
                if len(parts) < expected:
                    # May span multiple lines or have fewer fields
                    continue

                elem_id = int(float(parts[0]))
                pump_col = int(float(parts[1]))
                pump_frac = float(parts[2])
                dist_method = int(float(parts[3]))

                # Layer factors (NLayers values)
                layer_factors = [float(parts[4 + i]) for i in range(n_layers)]

                offset = 4 + n_layers
                dest_type = int(float(parts[offset]))
                dest_id = int(float(parts[offset + 1]))
                irig_col = int(float(parts[offset + 2]))
                adjust_col = int(float(parts[offset + 3]))
                pump_max_col = int(float(parts[offset + 4]))
                pump_max_frac = float(parts[offset + 5])

                config.elem_pumping_specs.append(
                    ElementPumpingSpec(
                        element_id=elem_id,
                        pump_column=pump_col,
                        pump_fraction=pump_frac,
                        dist_method=dist_method,
                        layer_factors=layer_factors,
                        dest_type=dest_type,
                        dest_id=dest_id,
                        irig_frac_column=irig_col,
                        adjust_column=adjust_col,
                        pump_max_column=pump_max_col,
                        pump_max_fraction=pump_max_frac,
                    )
                )

            # Read element groups
            config.elem_groups = self._read_element_groups(f)

    def _read_element_groups(self, f: TextIO) -> list[ElementGroup]:
        """Read element groups for pumping destinations."""
        groups: list[ElementGroup] = []

        n_groups_str = _next_data_or_empty(f)
        if not n_groups_str:
            return groups

        try:
            n_groups = int(n_groups_str)
        except ValueError:
            return groups

        for _ in range(n_groups):
            line = self._next_data_line(f)
            parts = line.split()
            if len(parts) < 3:
                continue

            group_id = int(parts[0])
            n_elems = int(parts[1])
            elements = [int(parts[2])]

            # Read remaining elements (if more than 1)
            for _ in range(n_elems - 1):
                elem_line = self._next_data_line(f)
                elements.append(int(elem_line.split()[0]))

            groups.append(ElementGroup(id=group_id, elements=elements))

        return groups

    def _read_version(self, f: TextIO) -> str:
        """Read the version header."""
        for line in f:
            self._line_num += 1
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped[1:].strip()
            if line[0] in COMMENT_CHARS:
                continue
            break
        return ""

    def _next_data_line(self, f: TextIO) -> str:
        """Return the next non-comment data line."""
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            return line.strip()
        raise FileFormatError("Unexpected end of file", line_number=self._line_num)


def read_gw_pumping(
    filepath: Path | str,
    base_dir: Path | None = None,
    n_layers: int = 1,
) -> PumpingConfig:
    """Read IWFM GW pumping file.

    Args:
        filepath: Path to the pumping main file
        base_dir: Base directory for resolving relative paths
        n_layers: Number of aquifer layers

    Returns:
        PumpingConfig with all pumping data
    """
    reader = PumpingReader()
    return reader.read(filepath, base_dir, n_layers)
