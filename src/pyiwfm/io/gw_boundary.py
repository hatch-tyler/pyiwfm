"""
Groundwater Boundary Conditions Reader for IWFM.

This module reads the IWFM groundwater boundary conditions file, which
contains four types of boundary conditions:
1. Specified Flow BCs (fixed flux at nodes)
2. Specified Head BCs (fixed head at nodes)
3. General Head BCs (head-dependent flow)
4. Constrained General Head BCs (head-dependent with constraints)

The main BC file is a dispatcher that references sub-files for each BC type,
plus an optional time series data file for dynamic BCs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pyiwfm.core.exceptions import FileFormatError
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
# Data classes for each BC type
# =============================================================================


@dataclass
class SpecifiedFlowBC:
    """Specified flow boundary condition at a node.

    Attributes:
        node_id: GW node ID
        layer: Aquifer layer (1-based)
        ts_column: Time series column (0 = static)
        base_flow: Base flow value (positive = inflow)
    """

    node_id: int
    layer: int
    ts_column: int = 0
    base_flow: float = 0.0


@dataclass
class SpecifiedHeadBC:
    """Specified head boundary condition at a node.

    Attributes:
        node_id: GW node ID
        layer: Aquifer layer (1-based)
        ts_column: Time series column (0 = static)
        head_value: Head value
    """

    node_id: int
    layer: int
    ts_column: int = 0
    head_value: float = 0.0


@dataclass
class GeneralHeadBC:
    """General head boundary condition at a node.

    Flow is computed as: Q = conductance * (external_head - gw_head)

    Attributes:
        node_id: GW node ID
        layer: Aquifer layer (1-based)
        ts_column: Time series column for external head (0 = static)
        external_head: External head value
        conductance: BC conductance
    """

    node_id: int
    layer: int
    ts_column: int = 0
    external_head: float = 0.0
    conductance: float = 0.0


@dataclass
class ConstrainedGeneralHeadBC:
    """Constrained general head boundary condition at a node.

    Like GeneralHeadBC but with a constraining head and maximum flow.

    Attributes:
        node_id: GW node ID
        layer: Aquifer layer (1-based)
        ts_column: Time series column for external head (0 = static)
        external_head: External head value
        conductance: BC conductance
        constraining_head: Head below which the constraining head is used
        max_flow_ts_column: Time series column for max flow (0 = static)
        max_flow: Maximum BC flow
    """

    node_id: int
    layer: int
    ts_column: int = 0
    external_head: float = 0.0
    conductance: float = 0.0
    constraining_head: float = 0.0
    max_flow_ts_column: int = 0
    max_flow: float = 0.0


@dataclass
class GWBoundaryConfig:
    """Complete GW boundary conditions configuration.

    Attributes:
        sp_flow_file: Path to specified flow BC sub-file
        sp_head_file: Path to specified head BC sub-file
        gh_file: Path to general head BC sub-file
        cgh_file: Path to constrained general head BC sub-file
        ts_data_file: Path to time series BC data file

        specified_flow_bcs: List of specified flow BCs
        sp_flow_factor: Conversion factor for specified flow
        sp_flow_time_unit: Time unit for specified flow

        specified_head_bcs: List of specified head BCs
        sp_head_factor: Conversion factor for specified head

        general_head_bcs: List of general head BCs
        gh_head_factor: Head conversion factor for general head
        gh_conductance_factor: Conductance conversion factor
        gh_time_unit: Time unit for general head

        constrained_gh_bcs: List of constrained general head BCs
        cgh_head_factor: Head conversion factor
        cgh_max_flow_factor: Max flow conversion factor
        cgh_head_time_unit: Time unit for head
        cgh_conductance_factor: Conductance conversion factor
        cgh_conductance_time_unit: Time unit for conductance
    """

    # Sub-file paths
    sp_flow_file: Path | None = None
    sp_head_file: Path | None = None
    gh_file: Path | None = None
    cgh_file: Path | None = None
    ts_data_file: Path | None = None

    # Specified flow BCs
    specified_flow_bcs: list[SpecifiedFlowBC] = field(default_factory=list)
    sp_flow_factor: float = 1.0
    sp_flow_time_unit: str = ""

    # Specified head BCs
    specified_head_bcs: list[SpecifiedHeadBC] = field(default_factory=list)
    sp_head_factor: float = 1.0

    # General head BCs
    general_head_bcs: list[GeneralHeadBC] = field(default_factory=list)
    gh_head_factor: float = 1.0
    gh_conductance_factor: float = 1.0
    gh_time_unit: str = ""

    # Constrained general head BCs
    constrained_gh_bcs: list[ConstrainedGeneralHeadBC] = field(default_factory=list)
    cgh_head_factor: float = 1.0
    cgh_max_flow_factor: float = 1.0
    cgh_head_time_unit: str = ""
    cgh_conductance_factor: float = 1.0
    cgh_conductance_time_unit: str = ""

    # Boundary node flow output section (NOUTB)
    n_bc_output_nodes: int = 0
    bc_output_file: Path | None = None
    bc_output_file_raw: str = ""  # Unresolved path string from file
    bc_output_specs: list[dict] = field(default_factory=list)  # [{id, layer, node, name}]

    @property
    def n_specified_flow(self) -> int:
        return len(self.specified_flow_bcs)

    @property
    def n_specified_head(self) -> int:
        return len(self.specified_head_bcs)

    @property
    def n_general_head(self) -> int:
        return len(self.general_head_bcs)

    @property
    def n_constrained_gh(self) -> int:
        return len(self.constrained_gh_bcs)

    @property
    def total_bcs(self) -> int:
        return (
            self.n_specified_flow
            + self.n_specified_head
            + self.n_general_head
            + self.n_constrained_gh
        )


class GWBoundaryReader:
    """Reader for IWFM groundwater boundary conditions files.

    The main BC file contains 5 lines of sub-file paths:
    1. Specified flow BC file
    2. Specified head BC file
    3. General head BC file
    4. Constrained general head BC file
    5. Time series BC data file

    Each sub-file has its own header with counts and conversion factors,
    followed by per-BC data rows.
    """

    def __init__(self) -> None:
        self._line_num = 0

    def read(self, filepath: Path | str, base_dir: Path | None = None) -> GWBoundaryConfig:
        """Read BC main file and all referenced sub-files.

        Args:
            filepath: Path to main BC file
            base_dir: Base directory for resolving relative paths

        Returns:
            GWBoundaryConfig with all BC data
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        config = GWBoundaryConfig()
        self._line_num = 0

        with open(filepath) as f:
            # Read 5 file paths
            sp_flow_path = _next_data_or_empty(f)
            if sp_flow_path:
                config.sp_flow_file = _resolve_path_f(base_dir, sp_flow_path)

            sp_head_path = _next_data_or_empty(f)
            if sp_head_path:
                config.sp_head_file = _resolve_path_f(base_dir, sp_head_path)

            gh_path = _next_data_or_empty(f)
            if gh_path:
                config.gh_file = _resolve_path_f(base_dir, gh_path)

            cgh_path = _next_data_or_empty(f)
            if cgh_path:
                config.cgh_file = _resolve_path_f(base_dir, cgh_path)

            ts_path = _next_data_or_empty(f)
            if ts_path:
                config.ts_data_file = _resolve_path_f(base_dir, ts_path)

            # NOUTB section (boundary node flow output)
            noutb_str = _next_data_or_empty(f)
            if noutb_str:
                try:
                    config.n_bc_output_nodes = int(noutb_str)
                except ValueError:
                    pass

            if config.n_bc_output_nodes > 0:
                # Output file path
                bhydout_path = _next_data_or_empty(f)
                if bhydout_path:
                    config.bc_output_file_raw = bhydout_path
                    config.bc_output_file = _resolve_path_f(base_dir, bhydout_path)
                # Read NOUTB rows: ID, LAYER, NODE, NAME
                for _ in range(config.n_bc_output_nodes):
                    line = self._next_data_line(f)
                    parts = line.split(maxsplit=3)
                    if parts:
                        try:
                            spec: dict = {"id": int(float(parts[0]))}
                            if len(parts) > 1:
                                spec["layer"] = int(float(parts[1]))
                            if len(parts) > 2:
                                spec["node"] = int(float(parts[2]))
                            if len(parts) > 3:
                                spec["name"] = parts[3].strip()
                            config.bc_output_specs.append(spec)
                        except ValueError:
                            pass

        # Read each sub-file
        if config.sp_flow_file and config.sp_flow_file.exists():
            self._read_specified_flow(config.sp_flow_file, config)

        if config.sp_head_file and config.sp_head_file.exists():
            self._read_specified_head(config.sp_head_file, config)

        if config.gh_file and config.gh_file.exists():
            self._read_general_head(config.gh_file, config)

        if config.cgh_file and config.cgh_file.exists():
            self._read_constrained_gh(config.cgh_file, config)

        return config

    def _read_specified_flow(self, filepath: Path, config: GWBoundaryConfig) -> None:
        """Read specified flow BC sub-file."""
        self._line_num = 0
        with open(filepath) as f:
            # NQB
            nqb_str = _next_data_or_empty(f)
            nqb = int(nqb_str) if nqb_str else 0
            if nqb <= 0:
                return

            # FACT
            fact_str = _next_data_or_empty(f)
            config.sp_flow_factor = float(fact_str) if fact_str else 1.0

            # TimeUnit
            config.sp_flow_time_unit = _next_data_or_empty(f)

            # Read NQB rows: NodeID, Layer, TSColumn, BaseFlow
            for _ in range(nqb):
                line = self._next_data_line(f)
                parts = line.split()
                if len(parts) < 4:
                    continue

                config.specified_flow_bcs.append(
                    SpecifiedFlowBC(
                        node_id=int(float(parts[0])),
                        layer=int(float(parts[1])),
                        ts_column=int(float(parts[2])),
                        base_flow=float(parts[3]) * config.sp_flow_factor,
                    )
                )

    def _read_specified_head(self, filepath: Path, config: GWBoundaryConfig) -> None:
        """Read specified head BC sub-file."""
        self._line_num = 0
        with open(filepath) as f:
            # NHB
            nhb_str = _next_data_or_empty(f)
            nhb = int(nhb_str) if nhb_str else 0
            if nhb <= 0:
                return

            # FACT
            fact_str = _next_data_or_empty(f)
            config.sp_head_factor = float(fact_str) if fact_str else 1.0

            # Read NHB rows: NodeID, Layer, TSColumn, HeadValue
            for _ in range(nhb):
                line = self._next_data_line(f)
                parts = line.split()
                if len(parts) < 4:
                    continue

                config.specified_head_bcs.append(
                    SpecifiedHeadBC(
                        node_id=int(float(parts[0])),
                        layer=int(float(parts[1])),
                        ts_column=int(float(parts[2])),
                        head_value=float(parts[3]) * config.sp_head_factor,
                    )
                )

    def _read_general_head(self, filepath: Path, config: GWBoundaryConfig) -> None:
        """Read general head BC sub-file."""
        self._line_num = 0
        with open(filepath) as f:
            # NGB
            ngb_str = _next_data_or_empty(f)
            ngb = int(ngb_str) if ngb_str else 0
            if ngb <= 0:
                return

            # FACTH
            facth_str = _next_data_or_empty(f)
            config.gh_head_factor = float(facth_str) if facth_str else 1.0

            # FACTC
            factc_str = _next_data_or_empty(f)
            config.gh_conductance_factor = float(factc_str) if factc_str else 1.0

            # TimeUnit
            config.gh_time_unit = _next_data_or_empty(f)

            # Read NGB rows: NodeID, Layer, TSColumn, ExternalHead, Conductance
            for _ in range(ngb):
                line = self._next_data_line(f)
                parts = line.split()
                if len(parts) < 5:
                    continue

                config.general_head_bcs.append(
                    GeneralHeadBC(
                        node_id=int(float(parts[0])),
                        layer=int(float(parts[1])),
                        ts_column=int(float(parts[2])),
                        external_head=float(parts[3]) * config.gh_head_factor,
                        conductance=float(parts[4]) * config.gh_conductance_factor,
                    )
                )

    def _read_constrained_gh(self, filepath: Path, config: GWBoundaryConfig) -> None:
        """Read constrained general head BC sub-file."""
        self._line_num = 0
        with open(filepath) as f:
            # NCGB
            ncgb_str = _next_data_or_empty(f)
            ncgb = int(ncgb_str) if ncgb_str else 0
            if ncgb <= 0:
                return

            # FACTH
            facth_str = _next_data_or_empty(f)
            config.cgh_head_factor = float(facth_str) if facth_str else 1.0

            # FACTVL (max flow factor)
            factvl_str = _next_data_or_empty(f)
            config.cgh_max_flow_factor = float(factvl_str) if factvl_str else 1.0

            # TimeUnit (for head)
            config.cgh_head_time_unit = _next_data_or_empty(f)

            # FACTC (conductance factor)
            factc_str = _next_data_or_empty(f)
            config.cgh_conductance_factor = float(factc_str) if factc_str else 1.0

            # TimeUnit (for conductance)
            config.cgh_conductance_time_unit = _next_data_or_empty(f)

            # Read NCGB rows: NodeID, Layer, TSCol, ExtHead, Conductance,
            #                  ConstrainingHead, MaxFlowTSCol, MaxFlow
            for _ in range(ncgb):
                line = self._next_data_line(f)
                parts = line.split()
                if len(parts) < 8:
                    continue

                config.constrained_gh_bcs.append(
                    ConstrainedGeneralHeadBC(
                        node_id=int(float(parts[0])),
                        layer=int(float(parts[1])),
                        ts_column=int(float(parts[2])),
                        external_head=float(parts[3]) * config.cgh_head_factor,
                        conductance=float(parts[4]) * config.cgh_conductance_factor,
                        constraining_head=float(parts[5]) * config.cgh_head_factor,
                        max_flow_ts_column=int(float(parts[6])),
                        max_flow=float(parts[7]) * config.cgh_max_flow_factor,
                    )
                )

    def _next_data_line(self, f: TextIO) -> str:
        """Return the next non-comment data line."""
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            return line.strip()
        raise FileFormatError("Unexpected end of file", line_number=self._line_num)


def read_gw_boundary(filepath: Path | str, base_dir: Path | None = None) -> GWBoundaryConfig:
    """Read IWFM GW boundary conditions file.

    Args:
        filepath: Path to the BC main file
        base_dir: Base directory for resolving relative paths

    Returns:
        GWBoundaryConfig with all boundary condition data
    """
    reader = GWBoundaryReader()
    return reader.read(filepath, base_dir)
