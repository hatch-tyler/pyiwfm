"""
Groundwater component main-file reader (hierarchical dispatcher).

The IWFM Groundwater component main file is a hierarchical dispatcher
that references sub-files for boundary conditions, pumping, tile
drains, and subsidence; carries hydrograph and face-flow output
specs; and embeds aquifer parameters / Kh anomalies / parametric
grids inline. This module hosts :class:`GWMainFileConfig` (the
parsed configuration) and :class:`GWMainFileReader` (the parser).

Split out of :mod:`pyiwfm.io.groundwater.reader` in v2.0; the
``reader`` module now contains only the per-file readers (wells,
initial heads, subsidence) and the legacy free-form
``GroundwaterWriter``. See ``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import numpy as np
from numpy.typing import NDArray

from pyiwfm.components.groundwater import AquiferParameters, HydrographLocation
from pyiwfm.io.ascii.reader import COMMENT_CHARS
from pyiwfm.io.ascii.reader import (
    is_comment_line as _is_comment_line,
)
from pyiwfm.io.ascii.reader import (
    next_data_or_empty as _next_data_or_empty,
)
from pyiwfm.io.ascii.reader import (
    resolve_path as _resolve_path_f,
)
from pyiwfm.io.ascii.reader import (
    strip_inline_comment as _strip_comment,
)
from pyiwfm.io.groundwater.reader import (
    FaceFlowSpec,
    KhAnomalyEntry,
    ParametricGridData,
)


@dataclass
class GWMainFileConfig:
    """
    Configuration parsed from GW component main file.

    The groundwater component main file is a dispatcher that references
    sub-files for boundary conditions, tile drains, pumping, and subsidence.
    It also contains inline hydrograph output location data.

    Attributes:
        version: File format version (e.g., "4.0")
        bc_file: Path to boundary conditions file
        tile_drain_file: Path to tile drains file
        pumping_file: Path to pumping file
        subsidence_file: Path to subsidence file
        overwrite_file: Path to parameter overwrite file (optional)
        head_output_factor: Conversion factor for head output
        head_output_unit: Unit string for head output
        volume_output_factor: Conversion factor for volume output
        volume_output_unit: Unit string for volume output
        debug_flag: Debug output flag
        coord_factor: Coordinate conversion factor for hydrographs
        hydrograph_output_file: Path to hydrograph output file
        hydrograph_locations: List of GW observation point locations
    """

    version: str = ""

    # Sub-file paths (all resolved to absolute)
    bc_file: Path | None = None
    tile_drain_file: Path | None = None
    pumping_file: Path | None = None
    subsidence_file: Path | None = None
    overwrite_file: Path | None = None

    # Raw (unresolved) path strings for roundtrip fidelity
    raw_paths: dict[str, str] = field(default_factory=dict)

    # Conversion factors
    head_output_factor: float = 1.0
    head_output_unit: str = "FEET"
    volume_output_factor: float = 1.0
    volume_output_unit: str = "TAF"
    velocity_output_factor: float = 1.0
    velocity_output_unit: str = ""

    # Output files
    velocity_output_file: Path | None = None
    vertical_flow_output_file: Path | None = None
    head_all_output_file: Path | None = None
    head_tecplot_file: Path | None = None
    velocity_tecplot_file: Path | None = None
    budget_output_file: Path | None = None
    zbudget_output_file: Path | None = None
    final_heads_file: Path | None = None

    # Debug and hydrograph output
    debug_flag: int = 0
    coord_factor: float = 1.0
    hydrograph_output_file: Path | None = None
    hydrograph_locations: list[HydrographLocation] = field(default_factory=list)

    # Element face flow output
    n_face_flow_outputs: int = 0
    face_flow_output_file: Path | None = None
    face_flow_specs: list[FaceFlowSpec] = field(default_factory=list)

    # Aquifer parameters
    aquifer_params: AquiferParameters | None = None

    # Aquifer parameter conversion factors and time units (for roundtrip)
    n_param_groups: int = 0  # NGROUP
    aq_factors_line: str = ""  # Raw FX FKH FS FN FV FL line
    aq_time_unit_kh: str = ""  # TUNITKH
    aq_time_unit_v: str = ""  # TUNITV (aquitard vertical K)
    aq_time_unit_l: str = ""  # TUNITL (aquifer vertical K)
    tecplot_print_flag: int = 1  # ITECPLOTFLAG

    # Kh anomaly overwrites (parsed but not yet applied to node arrays)
    kh_anomalies: list[KhAnomalyEntry] = field(default_factory=list)
    kh_anomaly_factor: float = 1.0  # FACT for Kh anomaly
    kh_anomaly_time_unit: str = ""  # TUNITH for Kh anomaly

    # Return flow section
    return_flow_flag: int = 0  # IFLAGRF

    # Parametric grid data (NGROUP > 0); interpolated later in model.py
    parametric_grids: list[ParametricGridData] = field(default_factory=list)

    # Initial heads
    initial_heads: NDArray[np.float64] | None = field(default=None, repr=False)


class GWMainFileReader:
    """
    Reader for IWFM groundwater component main file.

    The GW main file is a hierarchical dispatcher that contains:
    1. Version header (e.g., #4.0)
    2. Paths to sub-files (BC, tile drains, pumping, subsidence)
    3. Output conversion factors and units
    4. Inline hydrograph location data

    This reader parses the main file to extract configuration and
    hydrograph locations. It does NOT parse the sub-files - use
    the dedicated readers (e.g., GroundwaterReader.read_wells) for those.
    """

    def __init__(self) -> None:
        self._line_num = 0

    def read(self, filepath: Path | str, base_dir: Path | None = None) -> GWMainFileConfig:
        """
        Parse GW main file, extracting config and hydrograph locations.

        Args:
            filepath: Path to the GW component main file
            base_dir: Base directory for resolving relative paths.
                     If None, uses the parent directory of filepath.

        Returns:
            GWMainFileConfig with parsed values
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        config = GWMainFileConfig()
        self._line_num = 0

        with open(filepath, encoding="utf-8") as f:
            # Read version header (first non-comment line starting with #)
            config.version = self._read_version(f)

            # Read file paths sequentially:
            # BCFL (boundary conditions file)
            bc_path = _next_data_or_empty(f)
            if bc_path:
                config.raw_paths["bc"] = bc_path
                config.bc_file = _resolve_path_f(base_dir, bc_path)

            # TDFL (tile drains file)
            td_path = _next_data_or_empty(f)
            if td_path:
                config.raw_paths["td"] = td_path
                config.tile_drain_file = _resolve_path_f(base_dir, td_path)

            # PUMPFL (pumping file)
            pump_path = _next_data_or_empty(f)
            if pump_path:
                config.raw_paths["pump"] = pump_path
                config.pumping_file = _resolve_path_f(base_dir, pump_path)

            # SUBSFL (subsidence file)
            subs_path = _next_data_or_empty(f)
            if subs_path:
                config.raw_paths["subs"] = subs_path
                config.subsidence_file = _resolve_path_f(base_dir, subs_path)

            # OVRWRTFL (optional overwrite file, may be empty)
            ovr_path = _next_data_or_empty(f)
            if ovr_path:
                config.raw_paths["overwrite"] = ovr_path
                config.overwrite_file = _resolve_path_f(base_dir, ovr_path)

            # FACTLTOU (head output conversion factor)
            factltou = _next_data_or_empty(f)
            if factltou:
                try:
                    config.head_output_factor = float(factltou)
                except ValueError:
                    pass

            # UNITLTOU (head output unit)
            unitltou = _next_data_or_empty(f)
            if unitltou:
                config.head_output_unit = unitltou

            # FACTVLOU (volume output conversion factor)
            factvlou = _next_data_or_empty(f)
            if factvlou:
                try:
                    config.volume_output_factor = float(factvlou)
                except ValueError:
                    pass

            # UNITVLOU (volume output unit)
            unitvlou = _next_data_or_empty(f)
            if unitvlou:
                config.volume_output_unit = unitvlou

            # FACTVROU (velocity output factor)
            factvrou = _next_data_or_empty(f)
            if factvrou:
                try:
                    config.velocity_output_factor = float(factvrou)
                except ValueError:
                    pass

            # UNITVROU (velocity output unit)
            unitvrou = _next_data_or_empty(f)
            if unitvrou:
                config.velocity_output_unit = unitvrou

            # VELOUTFL (velocity output file - optional)
            vel_path = _next_data_or_empty(f)
            if vel_path:
                config.raw_paths["velocity"] = vel_path
                config.velocity_output_file = _resolve_path_f(base_dir, vel_path)

            # VFLOWOUTFL (vertical flow output file - optional)
            vflow_path = _next_data_or_empty(f)
            if vflow_path:
                config.raw_paths["vflow"] = vflow_path
                config.vertical_flow_output_file = _resolve_path_f(base_dir, vflow_path)

            # GWALLOUTFL (GW head all output file - optional)
            headall_path = _next_data_or_empty(f)
            if headall_path:
                config.raw_paths["headall"] = headall_path
                config.head_all_output_file = _resolve_path_f(base_dir, headall_path)

            # HTPOUTFL (TecPlot head output file - optional)
            htec_path = _next_data_or_empty(f)
            if htec_path:
                config.raw_paths["tecplot"] = htec_path
                config.head_tecplot_file = _resolve_path_f(base_dir, htec_path)

            # VTPOUTFL (TecPlot velocity output file - optional)
            vtec_path = _next_data_or_empty(f)
            if vtec_path:
                config.raw_paths["vtk"] = vtec_path
                config.velocity_tecplot_file = _resolve_path_f(base_dir, vtec_path)

            # GWBUDFL (GW budget output file - optional)
            bud_path = _next_data_or_empty(f)
            if bud_path:
                config.raw_paths["budget"] = bud_path
                config.budget_output_file = _resolve_path_f(base_dir, bud_path)

            # ZBUDFL (Zone budget output file - optional)
            zbud_path = _next_data_or_empty(f)
            if zbud_path:
                config.raw_paths["zbudget"] = zbud_path
                config.zbudget_output_file = _resolve_path_f(base_dir, zbud_path)

            # FNGWFL (final condition output file - optional)
            final_path = _next_data_or_empty(f)
            if final_path:
                config.raw_paths["final_heads"] = final_path
                config.final_heads_file = _resolve_path_f(base_dir, final_path)

            # iTecPlotFlag / KDEB / NOUTH — 3-value lookahead
            # iTecPlotFlag is optional in IWFM.  The Fortran determines its
            # presence by counting non-comment lines (20+ → flag present,
            # 19 → absent).  We detect it here by reading three values and
            # checking whether the third parses as int (NOUTH) or float
            # (FACTXY).  int(line_c) succeeds for NOUTH (an integer) but
            # fails for FACTXY (a float like "3.2808" or "1.0").
            line_a = _next_data_or_empty(f)
            line_b = _next_data_or_empty(f)
            line_c = _next_data_or_empty(f)

            # Detect: if line_c is a pure integer → (a=flag, b=KDEB, c=NOUTH)
            # If line_c contains '.' or fails int parse → (a=KDEB, b=NOUTH, c=FACTXY)
            has_tecplot_flag = False
            if line_c:
                try:
                    if "." not in line_c:
                        int(line_c)
                        has_tecplot_flag = True
                except ValueError:
                    pass

            if has_tecplot_flag:
                # a=iTecPlotFlag, b=KDEB, c=NOUTH
                if line_a:
                    try:
                        config.tecplot_print_flag = int(line_a)
                    except ValueError:
                        pass
                if line_b:
                    try:
                        config.debug_flag = int(line_b)
                    except ValueError:
                        pass
                nouth_str = line_c
                factxy = _next_data_or_empty(f)
                hydout_path = _next_data_or_empty(f)
            else:
                # a=KDEB, b=NOUTH, c=FACTXY (no iTecPlotFlag)
                if line_a:
                    try:
                        config.debug_flag = int(line_a)
                    except ValueError:
                        pass
                nouth_str = line_b
                factxy = line_c
                hydout_path = _next_data_or_empty(f)

            # Parse NOUTH
            n_hydrographs = 0
            if nouth_str:
                try:
                    n_hydrographs = int(nouth_str)
                except ValueError:
                    import logging

                    logging.getLogger(__name__).warning(
                        "Could not parse NOUTH value '%s' at line %d, assuming 0",
                        nouth_str,
                        self._line_num,
                    )

            # Parse FACTXY
            if factxy:
                try:
                    config.coord_factor = float(factxy)
                except ValueError:
                    pass

            # GWHYDOUTFL (hydrograph output file)
            if hydout_path:
                config.raw_paths["hydout"] = hydout_path
                config.hydrograph_output_file = _resolve_path_f(base_dir, hydout_path)

            # Read inline hydrograph location data (only if NOUTH > 0)
            if n_hydrographs > 0:
                config.hydrograph_locations = self._read_hydrograph_data(
                    f, n_hydrographs, config.coord_factor
                )

            # ── Element Face Flow Output ─────────────────────────────
            # NOUTF (number of element face flow hydrographs)
            noutf_str = _next_data_or_empty(f)
            if noutf_str:
                try:
                    config.n_face_flow_outputs = int(noutf_str)
                except ValueError:
                    pass

            # FCHYDOUTFL (face flow output file - optional)
            fc_path = _next_data_or_empty(f)
            if fc_path:
                config.raw_paths["faceflow"] = fc_path
                config.face_flow_output_file = _resolve_path_f(base_dir, fc_path)

            # Read inline face flow specifications (NOUTF rows)
            if config.n_face_flow_outputs > 0:
                config.face_flow_specs = self._read_face_flow_specs(f, config.n_face_flow_outputs)

            # ── Aquifer Parameters ───────────────────────────────────
            try:
                config.aquifer_params = self._read_aquifer_parameters(f, base_dir, config)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to read aquifer parameters at line %d: %s",
                    self._line_num,
                    exc,
                )

            # ── Anomaly in Hydraulic Conductivity ────────────────────
            try:
                config.kh_anomalies = self._read_kh_anomaly(f, config)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to read Kh anomalies at line %d: %s",
                    self._line_num,
                    exc,
                )

            # ── Return Flow ──────────────────────────────────────────
            try:
                iflagrf_str = _next_data_or_empty(f)
                if iflagrf_str:
                    config.return_flow_flag = int(float(iflagrf_str))
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to read return flow flag at line %d: %s",
                    self._line_num,
                    exc,
                )

            # ── Initial Heads ────────────────────────────────────────
            try:
                config.initial_heads = self._read_initial_heads(f)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to read initial heads at line %d: %s",
                    self._line_num,
                    exc,
                )

        return config

    def _read_version(self, f: TextIO) -> str:
        """Read the version header from the file."""
        for line in f:
            self._line_num += 1
            stripped = line.strip()
            if not stripped:
                continue
            # Version line starts with # followed by version number
            if stripped.startswith("#"):
                return stripped[1:].strip()
            # If we hit a comment line, continue
            if line[0] in COMMENT_CHARS:
                continue
            # If we hit data before version, there's no version header
            break
        return ""

    def _read_hydrograph_data(
        self, f: TextIO, n_hydrographs: int, coord_factor: float
    ) -> list[HydrographLocation]:
        """
        Read inline hydrograph location data.

        Format depends on HYDTYP:
        - HYDTYP=0 (x-y coords): ID  HYDTYP  IOUTHL  X  Y  NAME
        - HYDTYP=1 (node number): ID  HYDTYP  IOUTHL  IOUTH  NAME

        Args:
            f: Open file handle
            n_hydrographs: Number of hydrograph locations to read
            coord_factor: Coordinate conversion factor

        Returns:
            List of HydrographLocation objects
        """
        locations: list[HydrographLocation] = []
        count = 0

        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                # ID, HYDTYP, IOUTHL are always present
                int(parts[0])
                hydtyp = int(parts[1])
                layer = int(parts[2])

                if hydtyp == 0:
                    # x-y coordinates provided: ID HYDTYP IOUTHL X Y NAME
                    if len(parts) < 5:
                        continue
                    x = float(parts[3]) * coord_factor
                    y = float(parts[4]) * coord_factor
                    node_id = 0  # Will need to find nearest node
                    # Name is everything after the Y coordinate
                    name = " ".join(parts[5:]) if len(parts) > 5 else ""
                else:
                    # Node number provided: ID HYDTYP IOUTHL IOUTH NAME
                    x = 0.0
                    y = 0.0
                    node_id = int(parts[3])
                    # Name is everything after the node number
                    name = " ".join(parts[4:]) if len(parts) > 4 else ""

                locations.append(
                    HydrographLocation(node_id=node_id, layer=layer, x=x, y=y, name=name)
                )

                count += 1
                if count >= n_hydrographs:
                    break

            except (ValueError, IndexError):
                # Skip malformed lines
                continue

        return locations

    def _skip_data_lines(self, f: TextIO, count: int) -> None:
        """Skip *count* non-comment data lines."""
        skipped = 0
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            skipped += 1
            if skipped >= count:
                break

    def _read_face_flow_specs(self, f: TextIO, count: int) -> list[FaceFlowSpec]:
        """Read inline element face flow specifications.

        Format per line: ID  IOUTFL  IOUTFA  IOUTFB  NAME
        """
        specs: list[FaceFlowSpec] = []
        read_count = 0
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                spec = FaceFlowSpec(
                    id=int(parts[0]),
                    layer=int(parts[1]),
                    node_a=int(parts[2]),
                    node_b=int(parts[3]),
                    name=" ".join(parts[4:]) if len(parts) > 4 else "",
                )
                specs.append(spec)
            except (ValueError, IndexError):
                continue
            read_count += 1
            if read_count >= count:
                break
        return specs

    def _read_aquifer_parameters(
        self, f: TextIO, base_dir: Path, config: GWMainFileConfig | None = None
    ) -> AquiferParameters | None:
        """Read the inline Aquifer Parameters section.

        The section layout in the GW main file is:

        1. ``NGROUP`` — number of parametric grid groups.
           ``0`` means Option 2 (per-node parameters).
        2. Conversion factors (one data line):
           ``FX  FKH  FS  FN  FV  FL``
        3. Time units (three data lines):
           ``TUNITKH``, ``TUNITV``, ``TUNITL``
        4. If ``NGROUP > 0``: parametric grid definitions
           (stored in ``config.parametric_grids``).
        5. If ``NGROUP == 0``: per-node parameter data, one node per
           block of ``n_layers`` lines.  First line of each block has
           the node ID; continuation lines have parameters only.

        Per-node columns: ``PKH  PS  PN  PV  PL``
        (horizontal K, specific storage, specific yield,
        aquitard vertical K, aquifer vertical K)

        Returns ``None`` if the section cannot be parsed or if
        parametric grid mode is used (data stored on *config* instead).
        """
        # NGROUP
        ngroup_str = _next_data_or_empty(f)
        if not ngroup_str:
            return None
        try:
            ngroup = int(ngroup_str)
        except ValueError:
            return None

        # Store NGROUP on config for roundtrip
        if config is not None:
            config.n_param_groups = ngroup

        # Conversion factors: FX  FKH  FS  FN  FV  FL
        factors_str = _next_data_or_empty(f)
        if not factors_str:
            return None
        fparts = factors_str.split()
        if len(fparts) < 6:
            return None
        try:
            _fx = float(fparts[0])
            fkh = float(fparts[1])
            fs = float(fparts[2])
            fn = float(fparts[3])
            fv = float(fparts[4])
            fl = float(fparts[5])
        except ValueError:
            return None

        # Store raw factors line for roundtrip
        if config is not None:
            config.aq_factors_line = factors_str

        # Time units: TUNITKH, TUNITV, TUNITL
        tunitkh = _next_data_or_empty(f)
        tunitv = _next_data_or_empty(f)
        tunitl = _next_data_or_empty(f)
        if config is not None:
            config.aq_time_unit_kh = tunitkh
            config.aq_time_unit_v = tunitv
            config.aq_time_unit_l = tunitl

        if ngroup > 0:
            factors = (_fx, fkh, fs, fn, fv, fl)
            grids = self._read_parametric_aquifer_params(f, ngroup, factors)
            if config is not None:
                config.parametric_grids = grids
            return None

        # Option 2: per-node parameters
        # Read all node data blocks.  The first data line of each node
        # starts with the node ID followed by 5 parameter values.
        # Continuation lines (for layers 2..n_layers) have 5 values only.
        #
        # We don't know n_nodes or n_layers up front, so we collect
        # data dynamically and infer them.

        node_ids: list[int] = []
        # Per-node: list of (kh, ss, sy, aquitard_kv, kv) tuples per layer
        node_layers: list[list[tuple[float, float, float, float, float]]] = []

        current_node_data: list[tuple[float, float, float, float, float]] = []
        current_node_id: int | None = None

        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                if current_node_id is not None:
                    # A comment line after data started means end of
                    # aquifer params section.  Save last node and stop.
                    node_ids.append(current_node_id)
                    node_layers.append(current_node_data)
                    break
                # Haven't started reading data yet — skip comment.
                continue

            value, _ = _strip_comment(line)
            parts = value.split()
            if not parts:
                continue

            # Determine if this is a new node line or continuation.
            # A new node line has 6 fields: ID PKH PS PN PV PL
            # A continuation line has 5 fields: PKH PS PN PV PL
            # We distinguish by checking if the first value is an
            # integer that could be a node ID (and field count).
            if len(parts) == 6:
                # New node line
                try:
                    node_id = int(parts[0])
                    pkh = float(parts[1]) * fkh
                    ps = float(parts[2]) * fs
                    pn = float(parts[3]) * fn
                    pv = float(parts[4]) * fv  # aquitard vertical K
                    pl = float(parts[5]) * fl  # aquifer vertical K
                except ValueError:
                    break

                # Save previous node if any
                if current_node_id is not None:
                    node_ids.append(current_node_id)
                    node_layers.append(current_node_data)

                current_node_id = node_id
                current_node_data = [(pkh, ps, pn, pv, pl)]

            elif len(parts) == 5:
                # Continuation line for current node
                try:
                    pkh = float(parts[0]) * fkh
                    ps = float(parts[1]) * fs
                    pn = float(parts[2]) * fn
                    pv = float(parts[3]) * fv
                    pl = float(parts[4]) * fl
                except ValueError:
                    break
                current_node_data.append((pkh, ps, pn, pv, pl))

            else:
                # Unexpected format — end of section
                if current_node_id is not None:
                    node_ids.append(current_node_id)
                    node_layers.append(current_node_data)
                break

        if not node_ids:
            return None

        n_nodes = len(node_ids)
        n_layers = len(node_layers[0])

        # Build arrays (n_nodes, n_layers)
        kh = np.zeros((n_nodes, n_layers), dtype=np.float64)
        ss = np.zeros((n_nodes, n_layers), dtype=np.float64)
        sy = np.zeros((n_nodes, n_layers), dtype=np.float64)
        aquitard_kv = np.zeros((n_nodes, n_layers), dtype=np.float64)
        kv = np.zeros((n_nodes, n_layers), dtype=np.float64)

        for i, layers in enumerate(node_layers):
            for j, (h, s, y, av, v) in enumerate(layers):
                if j < n_layers:
                    kh[i, j] = h
                    ss[i, j] = s
                    sy[i, j] = y
                    aquitard_kv[i, j] = av
                    kv[i, j] = v

        return AquiferParameters(
            n_nodes=n_nodes,
            n_layers=n_layers,
            kh=kh,
            kv=kv,
            specific_storage=ss,
            specific_yield=sy,
            aquitard_kv=aquitard_kv,
        )

    def _read_parametric_aquifer_params(
        self,
        f: TextIO,
        ngroup: int,
        factors: tuple[float, float, float, float, float, float],
    ) -> list[ParametricGridData]:
        """Read NGROUP parametric grid definitions.

        Each parametric grid group contains:
        1. ``NDP NEP`` — number of parametric nodes and elements
        2. ``NEP`` element definition lines: ``ElemID  N1  N2  N3  N4``
        3. ``NDP`` node data lines:
           ``NodeID  X  Y  Param1_L1 ... Param5_LN``

        The 5 parameters per layer are: Kh, Ss, Sy, AquitardKv, Kv.

        Parameters
        ----------
        f : TextIO
            Open file handle positioned after the time-unit lines.
        ngroup : int
            Number of parametric grid groups to read.
        factors : tuple
            ``(FX, FKH, FS, FN, FV, FL)`` conversion factors.

        Returns
        -------
        list[ParametricGridData]
            One entry per parametric grid group.
        """
        fx, fkh, fs, fn, fv, fl = factors
        grids: list[ParametricGridData] = []

        for _ in range(ngroup):
            # IWFM parametric grid format per group:
            # 1. Node range string (e.g., "1-441")
            # 2. NDP (scalar)
            # 3. NEP (scalar)
            # 4. NEP element rows (if NEP > 0)
            # 5. NDP parametric node data rows

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

            # Read NEP element definitions (skip entirely when NEP=0)
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
            # First line per node: NodeID  X  Y  PKH  PS  PN  PV  PL  (8+ tokens)
            # Continuation lines (layers 2..NL): PKH  PS  PN  PV  PL  (5 tokens)
            node_coords = np.zeros((ndp, 2), dtype=np.float64)
            all_raw_values: list[list[float]] = []
            raw_node_lines: list[str] = []
            node_count = 0

            for line in f:
                self._line_num += 1
                if _is_comment_line(line):
                    if node_count >= ndp:
                        # All node first-lines read; a comment after means
                        # end of section (no more continuations).
                        break
                    continue
                value, _ = _strip_comment(line)
                nparts = value.split()
                if not nparts:
                    continue

                # Detect new node line vs continuation line.
                # A new node line has NodeID(int) X Y + 5 params = 8+ tokens.
                # A continuation line has exactly 5 float tokens.
                is_new_node = False
                if node_count < ndp and len(nparts) >= 8:
                    try:
                        _node_id = int(nparts[0])
                        float(nparts[1])
                        float(nparts[2])
                        is_new_node = True
                    except (ValueError, IndexError):
                        pass

                if is_new_node:
                    x_raw = float(nparts[1])
                    y_raw = float(nparts[2])
                    node_coords[node_count, 0] = x_raw * fx
                    node_coords[node_count, 1] = y_raw * fx
                    raw_vals = [float(v) for v in nparts[3:]]
                    all_raw_values.append(raw_vals)
                    raw_node_lines.append(value.strip())
                    node_count += 1
                elif len(nparts) == 5 and all_raw_values:
                    # Continuation line for current node
                    try:
                        cont_vals = [float(v) for v in nparts]
                        all_raw_values[-1].extend(cont_vals)
                        raw_node_lines.append(value.strip())
                    except ValueError:
                        break
                else:
                    # Not a node line or continuation — end of section.
                    # Use f.seek to restore position so the next reader
                    # can re-read this line.  Unfortunately, the line is
                    # already consumed from the iterator.  Instead, we
                    # simply break — the remaining sections will find
                    # their data from subsequent lines.
                    break

            if not all_raw_values:
                continue

            # Determine n_layers from value count:
            # 5 params * n_layers values per node
            n_values = len(all_raw_values[0])
            n_layers = n_values // 5 if n_values >= 5 else 1

            # Build node_values array: shape (ndp, n_layers, 5)
            # Raw layout per node: PKH_L1 PS_L1 PN_L1 PV_L1 PL_L1  PKH_L2 PS_L2 ...
            # i.e. all 5 params for layer 1, then all 5 for layer 2, etc.
            node_values = np.zeros((ndp, n_layers, 5), dtype=np.float64)
            param_factors = [fkh, fs, fn, fv, fl]
            for i, raw in enumerate(all_raw_values):
                for lay in range(n_layers):
                    for p in range(5):
                        idx = lay * 5 + p
                        if idx < len(raw):
                            node_values[i, lay, p] = raw[idx] * param_factors[p]

            grids.append(
                ParametricGridData(
                    n_nodes=ndp,
                    n_elements=nep,
                    elements=elements,
                    node_coords=node_coords[:node_count],
                    node_values=node_values[:node_count],
                    node_range_str=node_range_str,
                    raw_node_lines=raw_node_lines,
                )
            )

        return grids

    def _read_kh_anomaly(
        self, f: TextIO, config: GWMainFileConfig | None = None
    ) -> list[KhAnomalyEntry]:
        """Read the Anomaly in Hydraulic Conductivity section.

        Format::

            NEBK        (number of elements to overwrite, 0 = none)
            FACT        (conversion factor for anomaly K)
            TUNITH      (time unit string, e.g. 1DAY)
            IC  IEBK  BK[1]  BK[2] ... BK[n_layers]

        Returns a list of :class:`KhAnomalyEntry` objects with
        Kh values already multiplied by FACT.  The actual overwrite
        onto node arrays is performed later in ``model.py`` once the
        mesh element-to-node connectivity is available.
        """
        # NEBK
        nebk_str = _next_data_or_empty(f)
        if not nebk_str:
            return []
        try:
            nebk = int(nebk_str)
        except ValueError:
            return []

        # FACT (conversion factor) — always read, even when NEBK=0
        fact_str = _next_data_or_empty(f)
        try:
            fact = float(fact_str)
        except ValueError:
            fact = 1.0

        # TUNITH (time unit) — always read, even when NEBK=0
        tunith = _next_data_or_empty(f)

        # Store on config for roundtrip fidelity
        if config is not None:
            config.kh_anomaly_factor = fact
            config.kh_anomaly_time_unit = tunith

        if nebk <= 0:
            return []

        # Read NEBK anomaly data lines
        entries: list[KhAnomalyEntry] = []
        count = 0
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            value, _ = _strip_comment(line)
            parts = value.split()
            if len(parts) < 3:
                break
            try:
                # parts: IC, IEBK, BK[1], ..., BK[NLayers]
                element_id = int(parts[1])
                bk = [float(v) * fact for v in parts[2:]]
                entries.append(KhAnomalyEntry(element_id=element_id, kh_per_layer=bk))
            except (ValueError, IndexError):
                break
            count += 1
            if count >= nebk:
                break

        return entries

    def _read_initial_heads(self, f: TextIO) -> NDArray[np.float64] | None:
        """Read the Initial Groundwater Head Values section.

        Format::

            FACTHP          (conversion factor)
            ID  HP[1] HP[2] ... HP[n_layers]

        Returns an (n_nodes, n_layers) array, or None.
        """
        # FACTHP
        facthp_str = _next_data_or_empty(f)
        if not facthp_str:
            return None
        try:
            facthp = float(facthp_str)
        except ValueError:
            return None

        rows: list[list[float]] = []
        for line in f:
            self._line_num += 1
            if _is_comment_line(line):
                continue
            value, _ = _strip_comment(line)
            parts = value.split()
            if len(parts) < 2:
                break
            try:
                _node_id = int(parts[0])
                heads = [float(v) * facthp for v in parts[1:]]
                rows.append(heads)
            except ValueError:
                break

        if not rows:
            return None
        return np.array(rows, dtype=np.float64)


# Convenience functions


def read_gw_main_file(filepath: Path | str, base_dir: Path | None = None) -> GWMainFileConfig:
    """
    Read IWFM groundwater component main file.

    The GW main file is a hierarchical dispatcher that contains paths to
    sub-files (boundary conditions, pumping, etc.) and inline hydrograph
    location data.

    Args:
        filepath: Path to the GW component main file
        base_dir: Base directory for resolving relative paths.
                 If None, uses the parent directory of filepath.

    Returns:
        GWMainFileConfig with parsed values including hydrograph locations

    Example:
        >>> config = read_gw_main_file("C2VSimFG_Groundwater.dat")
        >>> print(f"Version: {config.version}")
        >>> print(f"Hydrograph locations: {len(config.hydrograph_locations)}")
        >>> if config.pumping_file:
        ...     wells = read_wells(config.pumping_file)
    """
    reader = GWMainFileReader()
    return reader.read(filepath, base_dir)
