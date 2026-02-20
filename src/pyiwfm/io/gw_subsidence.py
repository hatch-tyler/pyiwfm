"""
Groundwater Subsidence Reader for IWFM.

This module reads the IWFM subsidence parameter files, which define
compaction-related parameters for each aquifer node-layer. Two versions
are supported:

- Version 4.0: Basic subsidence with elastic/inelastic storage, interbed
  thickness, and pre-compaction head.
- Version 5.0: Enhanced with vertical interbed conductivity, number of
  equivalent delay interbeds, and preferred discretization thickness.
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
from pyiwfm.io.iwfm_reader import (
    strip_inline_comment as _strip_comment,
)


@dataclass
class SubsidenceHydrographSpec:
    """Subsidence hydrograph output location.

    Attributes:
        id: Hydrograph output ID (1-based)
        hydtyp: Hydrograph type (0=x-y coord, 1=node number)
        layer: Output layer
        x: X coordinate (if hydtyp=0)
        y: Y coordinate (if hydtyp=0)
        name: Optional name/description
    """

    id: int = 0
    hydtyp: int = 0
    layer: int = 1
    x: float = 0.0
    y: float = 0.0
    name: str = ""


@dataclass
class SubsidenceNodeParams:
    """Subsidence parameters for one node across all layers.

    Attributes:
        node_id: GW node ID
        elastic_sc: Elastic storage coefficient per layer
        inelastic_sc: Inelastic storage coefficient per layer
        interbed_thick: Interbed thickness per layer
        interbed_thick_min: Minimum interbed thickness per layer
        precompact_head: Pre-compaction head per layer
        kv_sub: Vertical conductivity of interbeds per layer (v5.0 only)
        n_eq: Number of equivalent delay interbeds per layer (v5.0 only)
    """

    node_id: int = 0
    elastic_sc: list[float] = field(default_factory=list)
    inelastic_sc: list[float] = field(default_factory=list)
    interbed_thick: list[float] = field(default_factory=list)
    interbed_thick_min: list[float] = field(default_factory=list)
    precompact_head: list[float] = field(default_factory=list)
    kv_sub: list[float] = field(default_factory=list)  # v5.0 only
    n_eq: list[float] = field(default_factory=list)  # v5.0 only


@dataclass
class ParametricSubsidenceData:
    """Raw parametric grid data for subsidence parameters.

    Attributes:
        n_nodes: Number of parametric grid nodes.
        n_elements: Number of parametric grid elements.
        elements: Element vertex index tuples (0-based into node arrays).
        node_coords: Parametric node coordinates, shape (n_nodes, 2).
        node_values: Parameter values per node, shape (n_nodes, n_layers, n_params).
            The 5 params are: ElasticSC, InelasticSC, InterbedThick, InterbedThickMin,
            PreCompactHead.
    """

    n_nodes: int
    n_elements: int
    elements: list[tuple[int, ...]]
    node_coords: NDArray[np.float64]
    node_values: NDArray[np.float64]


@dataclass
class SubsidenceConfig:
    """Complete subsidence configuration.

    Attributes:
        version: File format version (4.0 or 5.0)
        ic_file: Path to initial conditions file
        tecplot_file: Path to Tecplot output file
        final_subs_file: Path to end-of-simulation output file
        output_factor: Output unit conversion factor
        output_unit: Output unit string
        interbed_dz: Preferred interbed discretization thickness (v5.0 only)
        n_parametric_grids: Number of parametric grids (0 = direct input)
        conversion_factors: Conversion factor array (6 for v4.0, 7 for v5.0)

        node_params: List of SubsidenceNodeParams (one per node)
        n_nodes: Number of nodes with subsidence data
        n_layers: Number of aquifer layers

        ic_factor: IC file conversion factor
        ic_interbed_thick: IC interbed thickness array (n_nodes, n_layers)
        ic_precompact_head: IC pre-compaction head array (n_nodes, n_layers)
    """

    version: str = ""
    ic_file: Path | None = None
    tecplot_file: Path | None = None
    final_subs_file: Path | None = None
    output_factor: float = 1.0
    output_unit: str = "FEET"
    interbed_dz: float = 0.0  # v5.0 only
    n_parametric_grids: int = 0
    conversion_factors: list[float] = field(default_factory=list)

    # Hydrograph output section (NOUTS)
    n_hydrograph_outputs: int = 0
    hydrograph_coord_factor: float = 1.0
    hydrograph_output_file: Path | None = None
    hydrograph_specs: list[SubsidenceHydrographSpec] = field(default_factory=list)

    parametric_grids: list[ParametricSubsidenceData] = field(default_factory=list)

    node_params: list[SubsidenceNodeParams] = field(default_factory=list)
    n_nodes: int = 0
    n_layers: int = 0

    # Initial conditions
    ic_factor: float = 1.0
    ic_interbed_thick: NDArray[np.float64] | None = None
    ic_precompact_head: NDArray[np.float64] | None = None


class SubsidenceReader:
    """Reader for IWFM subsidence parameter files.

    Supports both version 4.0 and 5.0 formats. Version is auto-detected
    from the file header.
    """

    def __init__(self) -> None:
        self._line_num = 0

    def read(
        self,
        filepath: Path | str,
        base_dir: Path | None = None,
        n_nodes: int = 0,
        n_layers: int = 0,
    ) -> SubsidenceConfig:
        """Read subsidence parameter file.

        Args:
            filepath: Path to the subsidence file
            base_dir: Base directory for resolving relative paths
            n_nodes: Number of GW nodes (needed for reading parameters)
            n_layers: Number of aquifer layers

        Returns:
            SubsidenceConfig with all subsidence data
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        config = SubsidenceConfig()
        config.n_nodes = n_nodes
        config.n_layers = n_layers
        self._line_num = 0

        with open(filepath) as f:
            # Version header
            config.version = self._read_version(f)

            # IC file
            ic_path = _next_data_or_empty(f)
            if ic_path:
                config.ic_file = _resolve_path_f(base_dir, ic_path)

            # Tecplot output file
            tec_path = _next_data_or_empty(f)
            if tec_path:
                config.tecplot_file = _resolve_path_f(base_dir, tec_path)

            # Final subsidence output file
            final_path = _next_data_or_empty(f)
            if final_path:
                config.final_subs_file = _resolve_path_f(base_dir, final_path)

            # Output conversion factor
            factor_str = _next_data_or_empty(f)
            if factor_str:
                config.output_factor = float(factor_str)

            # Output unit
            config.output_unit = _next_data_or_empty(f)

            # Hydrograph output section: NOUTS, then if NOUTS>0: FACTXY, SUBHYDOUTFL, rows
            nouts_str = _next_data_or_empty(f)
            if nouts_str:
                try:
                    config.n_hydrograph_outputs = int(nouts_str)
                except ValueError:
                    config.n_hydrograph_outputs = 0

            if config.n_hydrograph_outputs > 0:
                # FACTXY (coordinate conversion factor)
                factxy_str = _next_data_or_empty(f)
                if factxy_str:
                    try:
                        config.hydrograph_coord_factor = float(factxy_str)
                    except ValueError:
                        pass

                # SUBHYDOUTFL (output file path)
                hydout_path = _next_data_or_empty(f)
                if hydout_path:
                    config.hydrograph_output_file = _resolve_path_f(base_dir, hydout_path)

                # Read NOUTS rows:
                #   SUBTYP=0: ID SUBTYP IOUTSL X Y NAME
                #   SUBTYP=1: ID SUBTYP IOUTSL IOUTS NAME
                for _ in range(config.n_hydrograph_outputs):
                    line = self._next_data_line(f)
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    hydtyp = int(float(parts[1]))
                    spec = SubsidenceHydrographSpec(
                        id=int(float(parts[0])),
                        hydtyp=hydtyp,
                        layer=int(float(parts[2])),
                    )
                    if hydtyp == 0 and len(parts) >= 5:
                        # X-Y coordinate format
                        spec.x = float(parts[3]) * config.hydrograph_coord_factor
                        spec.y = float(parts[4]) * config.hydrograph_coord_factor
                        if len(parts) > 5:
                            spec.name = " ".join(parts[5:])
                    else:
                        # Node number format (SUBTYP=1)
                        spec.x = float(parts[3])  # node ID stored as x
                        if len(parts) > 4:
                            spec.name = " ".join(parts[4:])
                    # Name from / delimiter overrides
                    if "/" in line:
                        name_start = line.index("/") + 1
                        spec.name = line[name_start:].strip()
                    config.hydrograph_specs.append(spec)
            # When NOUTS=0, Fortran does NOT read FACTXY/SUBHYDOUTFL

            # v5.0 has interbed DZ before NGroup
            is_v50 = config.version.startswith("5")
            if is_v50:
                dz_str = _next_data_or_empty(f)
                if dz_str:
                    config.interbed_dz = float(dz_str)

            # Number of parametric grids
            ngroup_str = _next_data_or_empty(f)
            if ngroup_str:
                config.n_parametric_grids = int(ngroup_str)

            # Conversion factors (6 for v4.0, 7 for v5.0)
            n_factors = 7 if is_v50 else 6
            factors_str = _next_data_or_empty(f)
            if factors_str:
                config.conversion_factors = [float(x) for x in factors_str.split()]
                # If not enough on one line, read more
                while len(config.conversion_factors) < n_factors:
                    more = _next_data_or_empty(f)
                    if more:
                        config.conversion_factors.extend(float(x) for x in more.split())

            # Read parameter data
            if config.n_parametric_grids == 0 and n_nodes > 0 and n_layers > 0:
                self._read_direct_params(f, config, is_v50)
            elif config.n_parametric_grids > 0:
                config.parametric_grids = self._read_parametric_params(
                    f, config.n_parametric_grids, config.conversion_factors
                )

        # Read initial conditions file
        if config.ic_file and config.ic_file.exists() and n_nodes > 0:
            self._read_ic_file(config.ic_file, config)

        return config

    def _read_direct_params(self, f: TextIO, config: SubsidenceConfig, is_v50: bool) -> None:
        """Read direct parameter input (NGroup == 0).

        For each node, reads n_layers rows of subsidence parameters.
        First layer row includes node ID; subsequent layers have params only.
        """
        n_nodes = config.n_nodes
        n_layers = config.n_layers
        factors = config.conversion_factors

        for _ in range(n_nodes):
            node_params = SubsidenceNodeParams()

            for layer_idx in range(n_layers):
                line = self._next_data_line(f)
                parts = line.split()

                if layer_idx == 0:
                    # First layer includes node ID
                    node_params.node_id = int(float(parts[0]))
                    offset = 1
                else:
                    offset = 0

                if is_v50:
                    # v5.0: ElasticSC, InelasticSC, InterbedThick, InterbedThickMin,
                    #        PreCompactHead, Kvsub, NEQ
                    elastic = float(parts[offset]) * (factors[1] if len(factors) > 1 else 1.0)
                    inelastic = float(parts[offset + 1]) * (factors[2] if len(factors) > 2 else 1.0)
                    thick = float(parts[offset + 2]) * (factors[3] if len(factors) > 3 else 1.0)
                    thick_min = float(parts[offset + 3]) * (factors[4] if len(factors) > 4 else 1.0)
                    precompact = float(parts[offset + 4]) * (
                        factors[5] if len(factors) > 5 else 1.0
                    )
                    kv = float(parts[offset + 5]) * (factors[6] if len(factors) > 6 else 1.0)
                    neq = float(parts[offset + 6])

                    node_params.elastic_sc.append(elastic)
                    node_params.inelastic_sc.append(inelastic)
                    node_params.interbed_thick.append(thick)
                    node_params.interbed_thick_min.append(thick_min)
                    node_params.precompact_head.append(precompact)
                    node_params.kv_sub.append(kv)
                    node_params.n_eq.append(neq)
                else:
                    # v4.0: ElasticSC, InelasticSC, InterbedThick, InterbedThickMin,
                    #        PreCompactHead
                    elastic = float(parts[offset]) * (factors[1] if len(factors) > 1 else 1.0)
                    inelastic = float(parts[offset + 1]) * (factors[2] if len(factors) > 2 else 1.0)
                    thick = float(parts[offset + 2]) * (factors[3] if len(factors) > 3 else 1.0)
                    thick_min = float(parts[offset + 3]) * (factors[4] if len(factors) > 4 else 1.0)
                    precompact = float(parts[offset + 4]) * (
                        factors[5] if len(factors) > 5 else 1.0
                    )

                    node_params.elastic_sc.append(elastic)
                    node_params.inelastic_sc.append(inelastic)
                    node_params.interbed_thick.append(thick)
                    node_params.interbed_thick_min.append(thick_min)
                    node_params.precompact_head.append(precompact)

            config.node_params.append(node_params)

    def _read_parametric_params(
        self,
        f: TextIO,
        ngroup: int,
        factors: list[float],
    ) -> list[ParametricSubsidenceData]:
        """Read NGROUP parametric grid definitions for subsidence.

        Each parametric grid group contains:
        1. Node range string (e.g., "1-441")
        2. NDP (number of parametric nodes)
        3. NEP (number of parametric elements)
        4. NEP element definition lines (if NEP > 0)
        5. NDP node data lines with continuation rows per layer

        The 5 subsidence params per layer are:
        ElasticSC, InelasticSC, InterbedThick, InterbedThickMin, PreCompactHead.
        """
        # Conversion factors: [0]=FX, [1..5]=param factors
        fx = factors[0] if len(factors) > 0 else 1.0
        param_factors = [factors[i] if i < len(factors) else 1.0 for i in range(1, 6)]

        grids: list[ParametricSubsidenceData] = []

        for _ in range(ngroup):
            # Node range string
            node_range_str = _next_data_or_empty(f)
            if not node_range_str:
                break

            # NDP
            ndp_str = _next_data_or_empty(f)
            if not ndp_str:
                break
            try:
                ndp = int(ndp_str)
            except ValueError:
                break

            # NEP
            nep_str = _next_data_or_empty(f)
            if not nep_str:
                break
            try:
                nep = int(nep_str)
            except ValueError:
                break

            # Read NEP element definitions
            elements: list[tuple[int, ...]] = []
            if nep > 0:
                elem_count = 0
                for line in f:
                    self._line_num += 1
                    if _is_comment_line(line):
                        continue
                    value, _ = _strip_comment(line)
                    eparts = value.split()
                    if len(eparts) < 4:
                        break
                    try:
                        verts = [int(p) - 1 for p in eparts[1:]]
                        verts = [v for v in verts if v >= 0]
                        elements.append(tuple(verts))
                    except ValueError:
                        break
                    elem_count += 1
                    if elem_count >= nep:
                        break

            # Read NDP parametric node data lines
            # First line per node: NodeID X Y P1_L1 P2_L1 P3_L1 P4_L1 P5_L1 (8+ tokens)
            # Continuation lines (layers 2..NL): P1 P2 P3 P4 P5 (5 tokens)
            node_coords = np.zeros((ndp, 2), dtype=np.float64)
            all_raw_values: list[list[float]] = []
            node_count = 0

            for line in f:
                self._line_num += 1
                if _is_comment_line(line):
                    if node_count >= ndp:
                        break
                    continue
                value, _ = _strip_comment(line)
                nparts = value.split()
                if not nparts:
                    continue

                # Detect new node line vs continuation line
                is_new_node = False
                if node_count < ndp and len(nparts) >= 8:
                    try:
                        int(nparts[0])
                        float(nparts[1])
                        float(nparts[2])
                        is_new_node = True
                    except (ValueError, IndexError):
                        pass

                if is_new_node:
                    node_coords[node_count, 0] = float(nparts[1]) * fx
                    node_coords[node_count, 1] = float(nparts[2]) * fx
                    raw_vals = [float(v) for v in nparts[3:]]
                    all_raw_values.append(raw_vals)
                    node_count += 1
                elif len(nparts) == 5 and all_raw_values:
                    try:
                        cont_vals = [float(v) for v in nparts]
                        all_raw_values[-1].extend(cont_vals)
                    except ValueError:
                        break
                else:
                    break

            if not all_raw_values:
                continue

            # Determine n_layers from value count (5 params * n_layers)
            n_values = len(all_raw_values[0])
            n_layers = n_values // 5 if n_values >= 5 else 1

            # Build node_values array: shape (ndp, n_layers, 5)
            node_values = np.zeros((ndp, n_layers, 5), dtype=np.float64)
            for i, raw in enumerate(all_raw_values):
                for lay in range(n_layers):
                    for p in range(5):
                        idx = lay * 5 + p
                        if idx < len(raw):
                            node_values[i, lay, p] = raw[idx] * param_factors[p]

            grids.append(
                ParametricSubsidenceData(
                    n_nodes=node_count,
                    n_elements=nep,
                    elements=elements,
                    node_coords=node_coords[:node_count],
                    node_values=node_values[:node_count],
                )
            )

        return grids

    def _read_ic_file(self, filepath: Path, config: SubsidenceConfig) -> None:
        """Read subsidence initial conditions file.

        Format:
            Line 1: Conversion factor
            Per node: ID, InterbedThick_L1..Ln, PreCompactHead_L1..Ln
        """
        n_nodes = config.n_nodes
        n_layers = config.n_layers
        self._line_num = 0

        config.ic_interbed_thick = np.zeros((n_nodes, n_layers))
        config.ic_precompact_head = np.zeros((n_nodes, n_layers))

        with open(filepath) as f:
            # Conversion factor
            factor_str = _next_data_or_empty(f)
            config.ic_factor = float(factor_str) if factor_str else 1.0

            # Per node: ID + InterbedThick(NLayers) + PreCompactHead(NLayers)
            for i in range(n_nodes):
                line = self._next_data_line(f)
                parts = line.split()

                # Expected: ID + n_layers interbed thicknesses + n_layers precompact heads
                expected = 1 + 2 * n_layers
                if len(parts) < expected:
                    continue

                # node_id = int(float(parts[0]))  # 1-based
                for layer in range(n_layers):
                    config.ic_interbed_thick[i, layer] = float(parts[1 + layer]) * config.ic_factor
                    config.ic_precompact_head[i, layer] = (
                        float(parts[1 + n_layers + layer]) * config.ic_factor
                    )

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


def read_gw_subsidence(
    filepath: Path | str,
    base_dir: Path | None = None,
    n_nodes: int = 0,
    n_layers: int = 0,
) -> SubsidenceConfig:
    """Read IWFM GW subsidence parameter file.

    Args:
        filepath: Path to the subsidence file
        base_dir: Base directory for resolving relative paths
        n_nodes: Number of GW nodes
        n_layers: Number of aquifer layers

    Returns:
        SubsidenceConfig with all subsidence data
    """
    reader = SubsidenceReader()
    return reader.read(filepath, base_dir, n_nodes, n_layers)
