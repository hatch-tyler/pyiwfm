"""
Stream component main-file reader (hierarchical dispatcher).

The IWFM Stream component main file is a hierarchical dispatcher that
references sub-files for inflows, diversions, and bypasses, and
contains inline hydrograph output specifications. This module hosts
:class:`StreamMainFileConfig` (the parsed configuration) and
:class:`StreamMainFileReader` (the parser).

Split out of :mod:`pyiwfm.io.streams.reader` in v2.0; the ``reader``
module now contains only the per-file readers (stream nodes,
diversions) and the legacy free-form ``StreamWriter``. See
``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pyiwfm.io.ascii.reader import COMMENT_CHARS
from pyiwfm.io.ascii.reader import (
    is_comment_line as _is_comment_line,
)
from pyiwfm.io.ascii.reader import (
    next_data_or_empty as _next_data_or_empty,
)
from pyiwfm.io.ascii.reader import (
    parse_version as parse_stream_version,
)
from pyiwfm.io.ascii.reader import (
    resolve_path as _resolve_path_f,
)
from pyiwfm.io.ascii.reader import (
    version_ge as stream_version_ge,
)
from pyiwfm.io.streams.reader import (
    CrossSectionRow,
    StreamBedParamRow,
    StreamInitialConditionRow,
)


@dataclass
class StreamMainFileConfig:
    """
    Configuration parsed from Stream component main file.

    The stream component main file is a dispatcher that references
    sub-files for inflows, diversions, and bypasses. It also contains
    inline hydrograph output specifications.

    Attributes:
        version: File format version (e.g., "4.2")
        inflow_file: Path to stream inflow time series file
        diversion_spec_file: Path to diversion specifications file
        bypass_spec_file: Path to bypass specifications file
        diversion_file: Path to diversion time series file
        budget_output_file: Path to stream reach budget output
        diversion_budget_file: Path to diversion detail budget output
        hydrograph_count: Number of hydrograph output locations
        hydrograph_output_type: 0=flow, 1=stage, 2=both
        hydrograph_output_file: Path to hydrograph output file
        hydrograph_specs: List of (node_id, name) tuples for output locations
    """

    version: str = ""
    inflow_file: Path | None = None
    diversion_spec_file: Path | None = None
    bypass_spec_file: Path | None = None
    diversion_file: Path | None = None
    budget_output_file: Path | None = None
    diversion_budget_file: Path | None = None
    hydrograph_count: int = 0
    hydrograph_output_type: int = 0
    hydrograph_output_file: Path | None = None
    hydrograph_specs: list[tuple[int, str]] = field(default_factory=list)
    # v5.0 end-of-simulation flows file
    final_flow_file: Path | None = None
    # Hydrograph output factors
    hydrograph_flow_factor: float = 1.0
    hydrograph_flow_unit: str = ""
    hydrograph_elev_factor: float = 1.0
    hydrograph_elev_unit: str = ""
    # Stream node budget section
    node_budget_count: int = 0
    node_budget_output_file: Path | None = None
    node_budget_ids: list[int] = field(default_factory=list)
    # Stream bed parameters
    conductivity_factor: float = 1.0
    conductivity_time_unit: str = ""
    length_factor: float = 1.0
    bed_params: list[StreamBedParamRow] = field(default_factory=list)
    # Hydraulic disconnection
    interaction_type: int | None = None
    # Stream evaporation
    evap_area_file: Path | None = None
    evap_node_specs: list[tuple[int, int, int]] = field(default_factory=list)
    # v5.0 cross-section data
    roughness_factor: float = 1.0
    cross_section_length_factor: float = 1.0
    cross_section_data: list[CrossSectionRow] = field(default_factory=list)
    # v5.0 initial conditions
    ic_type: int = 0
    ic_time_unit: str = ""
    ic_factor: float = 1.0
    initial_conditions: list[StreamInitialConditionRow] = field(default_factory=list)


class StreamMainFileReader:
    """
    Reader for IWFM stream component main file.

    The Stream main file is a hierarchical dispatcher that contains:
    1. Version header (e.g., #4.2)
    2. Paths to sub-files (inflows, diversions, bypasses)
    3. Output file paths
    4. Inline hydrograph output specifications
    """

    def __init__(self) -> None:
        self._line_num = 0
        self._pushback_line: str | None = None

    def read(self, filepath: Path | str, base_dir: Path | None = None) -> StreamMainFileConfig:
        """
        Parse Stream main file, extracting config and hydrograph specs.

        Args:
            filepath: Path to the Stream component main file
            base_dir: Base directory for resolving relative paths.
                     If None, uses the parent directory of filepath.

        Returns:
            StreamMainFileConfig with parsed values
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        config = StreamMainFileConfig()
        self._line_num = 0
        self._pushback_line = None

        with open(filepath, encoding="utf-8") as f:
            # Read version header
            config.version = self._read_version(f)

            # INFLOWFL (inflow time series file)
            inflow_path = _next_data_or_empty(f)
            if inflow_path:
                config.inflow_file = _resolve_path_f(base_dir, inflow_path)

            # DIVSPECFL (diversion specification file)
            divspec_path = _next_data_or_empty(f)
            if divspec_path:
                config.diversion_spec_file = _resolve_path_f(base_dir, divspec_path)

            # BYPSPECFL (bypass specification file)
            bypspec_path = _next_data_or_empty(f)
            if bypspec_path:
                config.bypass_spec_file = _resolve_path_f(base_dir, bypspec_path)

            # DIVFL (diversion time series file)
            div_path = _next_data_or_empty(f)
            if div_path:
                config.diversion_file = _resolve_path_f(base_dir, div_path)

            # STRMRCHBUDFL (stream reach budget output file)
            budget_path = _next_data_or_empty(f)
            if budget_path:
                config.budget_output_file = _resolve_path_f(base_dir, budget_path)

            # DIVDTLBUDFL (diversion detail budget output file)
            divbud_path = _next_data_or_empty(f)
            if divbud_path:
                config.diversion_budget_file = _resolve_path_f(base_dir, divbud_path)

            # v5.0: end-of-simulation flows file (before hydrographs)
            if stream_version_ge(config.version, (5, 0)):
                final_flow_path = _next_data_or_empty(f)
                if final_flow_path:
                    config.final_flow_file = _resolve_path_f(base_dir, final_flow_path)

            # NOUTR (number of hydrograph output nodes)
            noutr_str = _next_data_or_empty(f)
            if not noutr_str:
                return config

            try:
                config.hydrograph_count = int(noutr_str)
            except ValueError:
                return config

            if config.hydrograph_count <= 0:
                # Still need to read remaining sections
                self._read_post_hydrograph_sections(f, config, base_dir)
                return config

            # IHSQR (hydrograph output type: 0=flow, 1=stage, 2=both)
            ihsqr = _next_data_or_empty(f)
            if ihsqr:
                try:
                    config.hydrograph_output_type = int(ihsqr)
                except ValueError:
                    pass

            # FACTSQOU (flow output conversion factor)
            factsqou = _next_data_or_empty(f)
            if factsqou:
                try:
                    config.hydrograph_flow_factor = float(factsqou)
                except ValueError:
                    pass

            # UNITSQOU (flow output units)
            config.hydrograph_flow_unit = _next_data_or_empty(f)

            # If stage output is included (type 1=stage, 2=both)
            if config.hydrograph_output_type in (1, 2):
                factltou = _next_data_or_empty(f)
                if factltou:
                    try:
                        config.hydrograph_elev_factor = float(factltou)
                    except ValueError:
                        pass
                config.hydrograph_elev_unit = _next_data_or_empty(f)

            # STRMHYDOUTFL (hydrograph output file)
            hydout_path = _next_data_or_empty(f)
            if hydout_path:
                config.hydrograph_output_file = _resolve_path_f(base_dir, hydout_path)

            # Read inline hydrograph output specifications
            config.hydrograph_specs = self._read_hydrograph_specs(f, config.hydrograph_count)

            # Read remaining sections after hydrographs
            self._read_post_hydrograph_sections(f, config, base_dir)

        return config

    def _read_post_hydrograph_sections(
        self, f: TextIO, config: StreamMainFileConfig, base_dir: Path
    ) -> None:
        """Read all sections that follow the hydrograph specifications."""
        # Stream node budget section
        self._read_node_budget_section(f, config, base_dir)

        # Stream bed parameters section (version-dependent columns)
        self._read_bed_params_section(f, config)

        # v5.0: cross-section data and initial conditions
        if stream_version_ge(config.version, (5, 0)):
            self._read_cross_section_data(f, config)
            self._read_initial_conditions(f, config)

        # Stream evaporation section (all versions)
        self._read_evaporation_section(f, config, base_dir)

    def _read_node_budget_section(
        self, f: TextIO, config: StreamMainFileConfig, base_dir: Path
    ) -> None:
        """Read stream node budget section: NBUDR, budget file, node IDs."""
        nbudr_str = self._maybe_read_pushback(f)
        if not nbudr_str:
            return
        try:
            config.node_budget_count = int(nbudr_str)
        except ValueError:
            return
        if config.node_budget_count <= 0:
            return
        # Budget output file
        bud_path = _next_data_or_empty(f)
        if bud_path:
            config.node_budget_output_file = _resolve_path_f(base_dir, bud_path)
        # Per-node IDs
        for _ in range(config.node_budget_count):
            node_str = _next_data_or_empty(f)
            if node_str:
                try:
                    config.node_budget_ids.append(int(node_str))
                except ValueError:
                    break

    def _read_bed_params_section(self, f: TextIO, config: StreamMainFileConfig) -> None:
        """Read stream bed parameters: FACTK, TUNITSK, FACTL, per-node rows.

        Column layout depends on version:
        v4.2:  IR  WETPR  IRGW  CSTRM  DSTRM  (5 columns)
        v4.0:  IR  CSTRM  DSTRM  WETPR         (4 columns)
        v4.1:  IR  CSTRM  DSTRM                (3 columns)
        v5.0:  IR  CSTRM  DSTRM                (3 columns)
        """
        # FACTK
        factk_str = self._maybe_read_pushback(f)
        if not factk_str:
            return
        try:
            config.conductivity_factor = float(factk_str)
        except ValueError:
            return

        # TUNITSK
        config.conductivity_time_unit = _next_data_or_empty(f)

        # FACTL
        factl_str = _next_data_or_empty(f)
        if factl_str:
            try:
                config.length_factor = float(factl_str)
            except ValueError:
                pass

        # Determine column layout based on version
        # v4.2 uses 5 columns; v5.0+ uses 3 columns (same as v4.1)
        version = parse_stream_version(config.version) if config.version else (4, 0)
        is_v42 = (4, 2) <= version < (5, 0)
        is_v40 = version < (4, 1)
        if is_v42:
            min_cols = 5  # IR, WETPR, IRGW, CSTRM, DSTRM
        elif is_v40:
            min_cols = 4  # IR, CSTRM, DSTRM, WETPR
        else:
            min_cols = 3  # IR, CSTRM, DSTRM

        # Per-node bed parameter rows
        # Auto-detect actual column count from first data row
        detected_ncols = None
        while True:
            line_val = _next_data_or_empty(f)
            if not line_val:
                break
            parts = line_val.split()

            # Auto-detect on first row: if 5 columns, treat as v4.2
            if detected_ncols is None and len(parts) >= 5:
                detected_ncols = len(parts)
                if not is_v42:
                    is_v42 = True
                    is_v40 = False
                    min_cols = 5
            elif detected_ncols is None:
                detected_ncols = len(parts)

            if len(parts) < min_cols:
                # Likely INTRCTYPE (1 column) — save for next read
                self._pushback_line = line_val
                break
            try:
                row = StreamBedParamRow(node_id=int(parts[0]))
                if is_v42:
                    # v4.2: IR, WETPR, IRGW, CSTRM, DSTRM
                    row.wetted_perimeter = float(parts[1])
                    row.gw_node = int(float(parts[2]))
                    row.conductivity = float(parts[3])
                    row.bed_thickness = float(parts[4])
                elif is_v40:
                    # v4.0: IR, CSTRM, DSTRM, WETPR
                    row.conductivity = float(parts[1])
                    row.bed_thickness = float(parts[2])
                    row.wetted_perimeter = float(parts[3])
                else:
                    # v4.1/v5.0: IR, CSTRM, DSTRM
                    row.conductivity = float(parts[1])
                    row.bed_thickness = float(parts[2])
                config.bed_params.append(row)
            except (ValueError, IndexError):
                self._pushback_line = line_val
                break

        # Read INTRCTYPE
        intrc_str = self._maybe_read_pushback(f)
        if intrc_str:
            try:
                config.interaction_type = int(intrc_str.split()[0])
            except (ValueError, IndexError):
                pass

    def _read_cross_section_data(self, f: TextIO, config: StreamMainFileConfig) -> None:
        """Read v5.0 cross-section data: FACTN, FACTLT, per-node 6-col rows."""
        # FACTN (roughness conversion factor)
        factn_str = self._maybe_read_pushback(f)
        if not factn_str:
            return
        try:
            config.roughness_factor = float(factn_str)
        except ValueError:
            return

        # FACTLT (length conversion factor for cross-section)
        factlt_str = _next_data_or_empty(f)
        if factlt_str:
            try:
                config.cross_section_length_factor = float(factlt_str)
            except ValueError:
                pass

        # Per-node cross-section rows (6 columns: IR BottomElev B0 s n MaxDepth)
        while True:
            line_val = _next_data_or_empty(f)
            if not line_val:
                break
            parts = line_val.split()
            if len(parts) < 6:
                self._pushback_line = line_val
                break
            try:
                row = CrossSectionRow(
                    node_id=int(parts[0]),
                    bottom_elev=float(parts[1]),
                    B0=float(parts[2]),
                    s=float(parts[3]),
                    n=float(parts[4]),
                    max_flow_depth=float(parts[5]),
                )
                config.cross_section_data.append(row)
            except (ValueError, IndexError):
                self._pushback_line = line_val
                break

    def _read_initial_conditions(self, f: TextIO, config: StreamMainFileConfig) -> None:
        """Read v5.0 initial conditions: ICType, TimeUnit, FACTH, per-node rows."""
        # IC Type
        ic_str = self._maybe_read_pushback(f)
        if not ic_str:
            return
        try:
            config.ic_type = int(ic_str.split()[0])
        except (ValueError, IndexError):
            return

        # Time unit (for flow IC)
        config.ic_time_unit = _next_data_or_empty(f)

        # FACTH (conversion factor)
        facth_str = _next_data_or_empty(f)
        if facth_str:
            try:
                config.ic_factor = float(facth_str)
            except ValueError:
                pass

        # Per-node IC rows (2 columns: IR value)
        while True:
            line_val = _next_data_or_empty(f)
            if not line_val:
                break
            parts = line_val.split()
            if len(parts) < 2:
                self._pushback_line = line_val
                break
            try:
                row = StreamInitialConditionRow(
                    node_id=int(parts[0]),
                    value=float(parts[1]),
                )
                config.initial_conditions.append(row)
            except (ValueError, IndexError):
                self._pushback_line = line_val
                break

    def _read_evaporation_section(
        self, f: TextIO, config: StreamMainFileConfig, base_dir: Path
    ) -> None:
        """Read stream evaporation: STARFL (area file), per-node 3-col rows."""
        # STARFL (stream surface area file)
        area_path = self._maybe_read_pushback(f)
        if area_path:
            config.evap_area_file = _resolve_path_f(base_dir, area_path)

        # Per-node evap specs (3 columns: IR ICETST ICARST)
        while True:
            line_val = _next_data_or_empty(f)
            if not line_val:
                break
            parts = line_val.split()
            if len(parts) < 3:
                break
            try:
                config.evap_node_specs.append(
                    (
                        int(parts[0]),
                        int(parts[1]),
                        int(parts[2]),
                    )
                )
            except (ValueError, IndexError):
                break

    def _maybe_read_pushback(self, f: TextIO) -> str:
        """Read pushback line if available, otherwise next data line."""
        if self._pushback_line is not None:
            val = self._pushback_line
            self._pushback_line = None
            return val
        return _next_data_or_empty(f)

    def _read_version(self, f: TextIO) -> str:
        """Read the version header from the file."""
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

    def _read_hydrograph_specs(self, f: TextIO, n_hydrographs: int) -> list[tuple[int, str]]:
        """
        Read inline hydrograph output specifications.

        Format per line: IOUTR  NAME
        - IOUTR: Stream node ID for output
        - NAME: Optional name/description

        Args:
            f: Open file handle
            n_hydrographs: Number of specifications to read

        Returns:
            List of (node_id, name) tuples
        """
        specs: list[tuple[int, str]] = []
        count = 0

        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue

            parts = line.split(None, 1)  # Split on first whitespace
            if not parts:
                continue

            try:
                node_id = int(parts[0])
                name = parts[1].strip() if len(parts) > 1 else ""
                specs.append((node_id, name))

                count += 1
                if count >= n_hydrographs:
                    break

            except ValueError:
                continue

        return specs


# Convenience function


def read_stream_main_file(
    filepath: Path | str, base_dir: Path | None = None
) -> StreamMainFileConfig:
    """
    Read IWFM stream component main file.

    The Stream main file is a hierarchical dispatcher that contains paths
    to sub-files (inflows, diversions, bypasses) and inline hydrograph
    output specifications.

    Args:
        filepath: Path to the Stream component main file
        base_dir: Base directory for resolving relative paths.
                 If None, uses the parent directory of filepath.

    Returns:
        StreamMainFileConfig with parsed values

    Example:
        >>> config = read_stream_main_file("C2VSimFG_Streams.dat")
        >>> print(f"Version: {config.version}")
        >>> print(f"Hydrograph outputs: {config.hydrograph_count}")
    """
    reader = StreamMainFileReader()
    return reader.read(filepath, base_dir)
